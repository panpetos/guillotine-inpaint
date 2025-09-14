# predict.py
# Minimal Stable Diffusion inpainting predictor for Cog/Replicate.
# - принимает URL'ы изображения и маски
# - маска: белое = ЗАКРАСИТЬ (inpaint), чёрное = оставить как есть
# - IP-Adapter подключается опционально; если пакет недоступен, работа не падает

from typing import Optional
import io
import os
import requests
from PIL import Image, ImageOps

import torch
from cog import BasePredictor, Input, Path

# ---------------- safe IP-Adapter import (optional) -----------------
try:
    from ip_adapter import IPAdapter                      # вариант пакета №1
except Exception:
    try:
        from ip_adapter.ip_adapter import IPAdapter       # вариант пакета №2
    except Exception as e:
        print(f"[warn] IP-Adapter not available: {e}")
        IPAdapter = None

if IPAdapter is None:
    class IPAdapter:  # type: ignore
        def __init__(self, *args, **kwargs): ...
        def __call__(self, *args, **kwargs): return None
# --------------------------------------------------------------------

# Diffusers
from diffusers import StableDiffusionInpaintPipeline


def _load_image_from_url(url: str) -> Image.Image:
    """Download image from URL into a PIL Image (RGB)."""
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    return img


def _load_mask_from_url(url: str, size_like: Optional[Image.Image] = None) -> Image.Image:
    """
    Download mask. Convert to single-channel 'L'.
    Expectation for SD-inpaint: WHITE=to inpaint, BLACK=keep.
    """
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    m = Image.open(io.BytesIO(r.content))
    # многие маски приходят с альфой — приведём к L:
    if m.mode in ("RGBA", "LA"):
        m = Image.alpha_composite(Image.new("RGBA", m.size, (0, 0, 0, 0)), m).split()[-1]  # взять альфу
        # альфа: белое=непрозрачное → обычно это «закрасить». Подойдёт.
        m = ImageOps.invert(m) if False else m  # здесь можно инвертнуть при другой конвенции
        m = m.convert("L")
    else:
        m = m.convert("L")

    if size_like is not None and m.size != size_like.size:
        m = m.resize(size_like.size, Image.NEAREST)
    return m


class Predictor(BasePredictor):
    def setup(self) -> None:
        """
        Загружаем пайплайн. Можно поменять модель через переменную окружения:
        SD_INPAINT_MODEL = "runwayml/stable-diffusion-inpainting"  (1.5)
                         или "stabilityai/stable-diffusion-2-inpainting"
        """
        model_id = os.getenv("SD_INPAINT_MODEL", "runwayml/stable-diffusion-inpainting")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        print(f"[setup] loading model '{model_id}' on {self.device} (dtype={dtype})")
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None,            # отключим safety checker, если есть
            requires_safety_checker=False,
        )
        self.pipe = self.pipe.to(self.device)

        # экономия памяти/видеоПамяти
        if self.device == "cuda":
            self.pipe.enable_attention_slicing()
            try:
                self.pipe.enable_vae_slicing()
            except Exception:
                pass

        # опционально инициализировать IP-Adapter (если он реально нужен в твоём коде)
        self.ip_adapter = None
        try:
            # Пример (заглушка). Если используешь — подставь свои веса/аргументы:
            # self.ip_adapter = IPAdapter(self.pipe, some_args...)
            pass
        except Exception as e:
            print(f"[warn] failed to init IP-Adapter: {e}")
            self.ip_adapter = None

    def predict(
        self,
        image: str = Input(
            description="URL исходного изображения (RGB).",
        ),
        mask: str = Input(
            description="URL маски: белое=затираем (inpaint), чёрное=оставляем.",
        ),
        prompt: str = Input(
            description="Текстовый промпт (необязательно). Пустой = просто залатать контекстом.",
            default="",
        ),
        negative_prompt: str = Input(
            description="Негативный промпт.",
            default="",
        ),
        guidance_scale: float = Input(
            description="Classifier-free guidance scale.",
            ge=0.0,
            le=20.0,
            default=7.5,
        ),
        num_inference_steps: int = Input(
            description="Число шагов диффузии.",
            ge=1,
            le=100,
            default=30,
        ),
        strength: float = Input(
            description="Сила инпейнта (0..1).",
            ge=0.0,
            le=1.0,
            default=0.85,
        ),
        seed: Optional[int] = Input(
            description="Фиксированный seed (None = случайный).",
            default=None,
        ),
    ) -> Path:
        # загрузим изображения
        base = _load_image_from_url(image)
        mask_img = _load_mask_from_url(mask, size_like=base)

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(seed))

        # основной вызов пайплайна
        out = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            image=base,
            mask_image=mask_img,
            guidance_scale=float(guidance_scale),
            num_inference_steps=int(num_inference_steps),
            strength=float(strength),
            generator=generator,
        ).images[0]

        out_path = Path("/tmp/out.png")
        out.save(str(out_path))
        return out_path
