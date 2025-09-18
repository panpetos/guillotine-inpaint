import torch
from cog import BasePredictor, Input, Path
from PIL import Image
import requests
from io import BytesIO
from diffusers import StableDiffusionInpaintPipeline
from typing import Union


def load_image_from_url(url: str) -> Image.Image:
    resp = requests.get(url)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert("RGB")


def load_image_any(src: Union[str, Path]) -> Image.Image:
    """Загрузка изображения: если строка (URL) → качаем, если файл → открываем напрямую"""
    if isinstance(src, Path):
        return Image.open(src).convert("RGB")
    elif isinstance(src, str):
        return load_image_from_url(src)
    else:
        raise ValueError("image/mask must be URL (str) or file (Path)")


class Predictor(BasePredictor):
    def setup(self):
        """Загрузка модели один раз при старте контейнера"""
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.float16,
        ).to("cuda")

    def predict(
        self,
        image: Union[str, Path] = Input(description="URL или файл исходного изображения"),
        mask: Union[str, Path] = Input(description="URL или файл маски (белое=затираем, чёрное=оставляем)"),
        prompt: str = Input(description="Текстовый промпт", default=""),
    ) -> Path:
        # Загружаем
        base_img = load_image_any(image)
        mask_img = load_image_any(mask).convert("L").resize(base_img.size)

        # Запускаем пайплайн
        result = self.pipe(
            prompt=prompt if prompt else "inpaint",
            image=base_img,
            mask_image=mask_img,
        ).images[0]

        out_path = Path("/tmp/out.png")
        result.save(out_path)
        return out_path
