from __future__ import annotations

import logging
import secrets
import threading
from io import BytesIO

from PIL import Image, ImageOps

from config import Settings


logger = logging.getLogger(__name__)


class QwenEngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._pipe = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._pipe is not None

    def _load(self):
        if self._pipe is not None:
            return self._pipe

        with self._load_lock:
            if self._pipe is not None:
                return self._pipe

            import torch
            from diffusers import QwenImage21Pipeline
            from diffusers.quantizers import PipelineQuantizationConfig
            from transformers import BitsAndBytesConfig

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA недоступна: Qwen Image требует NVIDIA GPU.")

            self.settings.hf_home.mkdir(parents=True, exist_ok=True)

            logger.info(
                "Loading Qwen Image model=%s gpu=%s",
                self.settings.model_id,
                torch.cuda.get_device_name(0),
            )

            # Точная схема загрузки, которую рекомендует model card этой 4-bit сборки:
            # transformer + Qwen3-VL text encoder в FP4, вычисления в BF16.
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="fp4",
                bnb_4bit_use_double_quant=False,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

            quant_config = PipelineQuantizationConfig(
                quant_mapping={
                    "transformer": bnb,
                    "text_encoder": bnb,
                }
            )

            pipe = QwenImage21Pipeline.from_pretrained(
                self.settings.model_id,
                dtype=torch.bfloat16,
                quantization_config=quant_config,
                cache_dir=str(self.settings.hf_home),
                token=self.settings.hf_token,
                low_cpu_mem_usage=True,
                local_files_only=False,
            )

            pipe.to("cuda")
            pipe.set_progress_bar_config(disable=True)

            # A10 (Ampere) поддерживает TF32/BF16.
            torch.backends.cuda.matmul.allow_tf32 = True

            self._pipe = pipe

            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(
                "Qwen Image loaded. CUDA allocated=%.2f GiB reserved=%.2f GiB",
                allocated,
                reserved,
            )
            return pipe

    def warmup(self) -> None:
        self._load()

    def _generator(self):
        import torch

        seed = secrets.randbelow(2_147_483_647)
        return torch.Generator(device="cuda").manual_seed(seed), seed

    @staticmethod
    def _fit_image(image: Image.Image, max_side: int) -> Image.Image:
        image = ImageOps.exif_transpose(image).convert("RGB")

        width, height = image.size
        longest = max(width, height)

        if longest > max_side:
            scale = max_side / longest
            width = max(256, int(width * scale))
            height = max(256, int(height * scale))

        # Qwen/VAE удобнее работать с размерами, кратными 16.
        width = max(256, round(width / 16) * 16)
        height = max(256, round(height / 16) * 16)

        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.LANCZOS)

        return image

    def generate(self, prompt: str) -> tuple[Image.Image, int]:
        pipe = self._load()

        with self._inference_lock:
            generator, seed = self._generator()

            image = pipe(
                prompt=prompt,
                width=self.settings.generate_width,
                height=self.settings.generate_height,
                num_inference_steps=self.settings.inference_steps,
                true_cfg_scale=self.settings.true_cfg_scale,
                generator=generator,
            ).images[0]

            return image, seed

    def edit(self, prompt: str, source: Image.Image) -> tuple[Image.Image, int]:
        pipe = self._load()
        source = self._fit_image(source, self.settings.edit_max_side)

        with self._inference_lock:
            generator, seed = self._generator()

            image = pipe(
                prompt=prompt,
                image=source,
                num_inference_steps=self.settings.inference_steps,
                true_cfg_scale=self.settings.true_cfg_scale,
                generator=generator,
            ).images[0]

            return image, seed

    @staticmethod
    def to_png_bytes(image: Image.Image) -> bytes:
        buffer = BytesIO()
        image.save(buffer, format="PNG", compress_level=6)
        return buffer.getvalue()

    def cuda_status(self) -> str:
        try:
            import torch

            if not torch.cuda.is_available():
                return "CUDA: недоступна"

            props = torch.cuda.get_device_properties(0)
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            total = props.total_memory / 1024**3

            return (
                f"GPU: {props.name}\n"
                f"VRAM: {allocated:.1f} GiB allocated / "
                f"{reserved:.1f} GiB reserved / {total:.1f} GiB total"
            )
        except Exception as exc:
            return f"GPU status error: {exc}"
