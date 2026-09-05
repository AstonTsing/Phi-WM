"""Frozen V-JEPA2 encoder wrapper for action/FDM training."""

from typing import List

import torch
from torch import nn

from deployment.model_server.tools.image_tools import to_pil_preserve


class VJEPA2Encoder(nn.Module):
    """Load a local HuggingFace V-JEPA2 checkpoint and return patch tokens."""

    def __init__(
        self,
        model_path: str,
        freeze: bool = True,
        torch_dtype: torch.dtype = torch.bfloat16,
        num_frames: int = 1,
    ):
        super().__init__()
        from transformers import AutoModel, AutoVideoProcessor

        self.processor = AutoVideoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=torch_dtype, local_files_only=True)
        self.num_frames = int(num_frames)
        if freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad_(False)

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    def _normalize_video_batch(self, images: List) -> List[List]:
        videos = []
        for image in images:
            if isinstance(image, (list, tuple)):
                frames = [to_pil_preserve(frame).convert("RGB") for frame in image]
            else:
                frame = to_pil_preserve(image).convert("RGB")
                frames = [frame for _ in range(max(1, self.num_frames))]
            if self.num_frames > 0:
                if len(frames) < self.num_frames:
                    frames = frames + [frames[-1] for _ in range(self.num_frames - len(frames))]
                elif len(frames) > self.num_frames:
                    frames = frames[: self.num_frames]
            videos.append(frames)
        return videos

    def forward(self, images: List) -> torch.Tensor:
        videos = self._normalize_video_batch(images)
        inputs = self.processor(videos=videos, return_tensors="pt")
        inputs = {
            key: value.to(device=self.device, dtype=self.dtype if value.is_floating_point() else value.dtype)
            for key, value in inputs.items()
        }
        use_grad = any(param.requires_grad for param in self.model.parameters())
        context = torch.enable_grad() if use_grad else torch.no_grad()
        with context:
            outputs = self.model(**inputs, return_dict=True)
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            return outputs.last_hidden_state
        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            return outputs.hidden_states[-1]
        if isinstance(outputs, (tuple, list)) and outputs:
            return outputs[0]
        raise RuntimeError("VJEPA2 model output does not contain hidden tokens")

