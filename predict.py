from cog import BasePredictor, Input, Path
from diffusers import StableDiffusionInpaintPipeline
from ip_adapter import IPAdapter
from PIL import Image

class Predictor(BasePredictor):
    def setup(self):
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype="float32"
        )
        self.pipe.to("cpu")  # если без GPU
        self.ip_adapter = IPAdapter(self.pipe)

    def predict(
        self,
        image: Path = Input(description="Основное фото дома"),
        mask: Path = Input(description="Маска для замены"),
        reference_image: Path = Input(description="Фото-референс окна"),
        prompt: str = Input(description="Текстовое описание результата"),
    ) -> Path:
        image = Image.open(image).convert("RGB")
        mask = Image.open(mask).convert("RGB")
        ref = Image.open(reference_image).convert("RGB")

        result = self.ip_adapter.generate(
            image=image,
            mask_image=mask,
            ip_adapter_image=ref,
            prompt=prompt,
            guidance_scale=7.5,
            num_inference_steps=30,
        )

        out_path = "/tmp/out.png"
        result[0].save(out_path)
        return Path(out_path)
