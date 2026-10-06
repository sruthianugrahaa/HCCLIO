"""inference.py - IoT tier: local ViT-Small/16 inference on the Raspberry Pi 4.

For one JPEG frame it returns
    p_i  softmax vector over the 1000 ImageNet-1k classes (float32)
    c_i  confidence  = max(p_i)
    y_i  prediction  = argmax(p_i)
    t_ms local inference time in ms (decode + preprocess + forward pass)

Model: timm `vit_small_patch16_224.augreg_in21k_ft_in1k` (ImageNet-21k pre-trained).
`backend="stub"` swaps in a simulated classifier so the pipeline can be tested
without torch.

    python inference.py some_image.jpg
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass

import numpy as np

MODEL_NAME = "vit_small_patch16_224.augreg_in21k_ft_in1k"


@dataclass
class LocalResult:
    p_i: np.ndarray
    c_i: float
    y_i: int
    t_ms: float


class ViTSmall:
    def __init__(self, model_name: str = MODEL_NAME, backend: str = "timm", threads: int = 4):
        self.backend = backend
        if backend == "stub":
            import sys
            from pathlib import Path

            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # outputs/ (holds common/)
            from common.stub_vit import StubClassifier

            self._stub = StubClassifier("iot", accuracy=0.75)
            return

        import timm
        import torch
        from PIL import Image

        torch.set_num_threads(threads)  # Pi 4 has 4 Cortex-A72 cores
        self.torch, self.Image = torch, Image
        self.model = timm.create_model(model_name, pretrained=True).eval()
        cfg = timm.data.resolve_data_config({}, model=self.model)
        self.transform = timm.data.create_transform(**cfg)
        # warm-up so frame 0's time is not inflated by lazy initialisation
        buf = io.BytesIO()
        Image.new("RGB", (224, 224)).save(buf, format="JPEG")
        self.infer(buf.getvalue())

    def infer(self, jpeg: bytes) -> LocalResult:
        if self.backend == "stub":
            p, t_ms = self._stub.predict(jpeg)
        else:
            t0 = time.perf_counter()
            img = self.Image.open(io.BytesIO(jpeg)).convert("RGB")
            x = self.transform(img).unsqueeze(0)
            with self.torch.inference_mode():
                p = self.torch.softmax(self.model(x)[0].float(), dim=-1).numpy().astype(np.float32)
            t_ms = (time.perf_counter() - t0) * 1000.0
        y = int(np.argmax(p))
        return LocalResult(p_i=p, c_i=float(p[y]), y_i=y, t_ms=t_ms)


if __name__ == "__main__":
    # Time ViT-Small on this device and suggest qoe.t_i_ms for common/hcclio.yaml:
    #     python inference.py some_frame.jpg [repeats]
    import sys

    import numpy as np

    model = ViTSmall()
    jpeg = open(sys.argv[1], "rb").read()
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    runs = [model.infer(jpeg) for _ in range(n)]
    t = np.array([r.t_ms for r in runs])
    print(f"y_i={runs[0].y_i}  c_i={runs[0].c_i:.3f}")
    print(f"ViT-Small over {n} runs: mean {t.mean():.1f} ms, median {np.median(t):.1f} ms, max {t.max():.1f} ms")
    print(f"suggested qoe.t_i_ms (2 x mean, same as calib_scale 2.0): {2 * t.mean():.0f}")
