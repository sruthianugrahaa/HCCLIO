"""model.py - Cloud tier: loads ViT-Large/16 (timm, ImageNet-21k pre-trained).

    vit_large_patch16_224.augreg_in21k_ft_in1k  ->  1000-class softmax p_CS

The model is loaded once at start-up and kept in memory (GPU if the laptop has
CUDA, else CPU). backend="stub" loads a simulated classifier for dry runs.

    python model.py           # load the model and time one forward pass
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # outputs/ (holds common/)

MODEL_NAME = "vit_large_patch16_224.augreg_in21k_ft_in1k"


class ViTLarge:
    """predict(jpeg) -> (p_CS softmax[1000] float32, inference time in ms)."""

    def __init__(self, model_name: str = MODEL_NAME, backend: str = "timm", accuracy: float = 0.85):
        self.backend = backend
        if backend == "stub":
            from common.stub_vit import StubClassifier

            self._stub = StubClassifier("cloud", accuracy)
            return
        import timm
        import torch
        from PIL import Image

        self.torch, self.Image = torch, Image
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = timm.create_model(model_name, pretrained=True).eval().to(self.device)
        cfg = timm.data.resolve_data_config({}, model=self.model)
        self.transform = timm.data.create_transform(**cfg)
        buf = io.BytesIO()
        Image.new("RGB", (224, 224)).save(buf, format="JPEG")
        self.predict(buf.getvalue())  # warm-up

    def predict(self, jpeg: bytes) -> tuple[np.ndarray, float]:
        if self.backend == "stub":
            return self._stub.predict(jpeg)
        t0 = time.perf_counter()
        img = self.Image.open(io.BytesIO(jpeg)).convert("RGB")
        x = self.transform(img).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            p = self.torch.softmax(self.model(x)[0].float(), dim=-1).cpu().numpy().astype(np.float32)
        if self.device == "cuda":
            self.torch.cuda.synchronize()
        return p, (time.perf_counter() - t0) * 1000.0


if __name__ == "__main__":
    m = ViTLarge()
    buf = io.BytesIO()
    m.Image.new("RGB", (224, 224)).save(buf, format="JPEG")
    p, ms = m.predict(buf.getvalue())
    print(f"{MODEL_NAME} on {m.device}: {ms:.1f} ms, argmax {int(p.argmax())}")
