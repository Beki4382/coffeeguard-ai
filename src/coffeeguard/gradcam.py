from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .config import TRAIN_CONFIG
from .transforms import IMAGENET_MEAN, IMAGENET_STD, eval_transform


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.fwd_handle = target_layer.register_forward_hook(self._forward_hook)
        self.bwd_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, _module, _input, output):
        self.activations = output.detach()

    def _backward_hook(self, _module, _grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()

    def __call__(self, image_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image_tensor)
        score = logits[:, class_idx].sum()
        score.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze()
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.detach().cpu().numpy()


def overlay_gradcam(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    import matplotlib.cm as cm

    base = image.convert("RGB").resize((TRAIN_CONFIG.image_size, TRAIN_CONFIG.image_size))
    heat = Image.fromarray(np.uint8(cm.jet(heatmap)[..., :3] * 255)).resize(base.size)
    return Image.blend(base, heat, alpha=alpha)


def tensor_from_image(image: Image.Image, device: torch.device) -> torch.Tensor:
    return eval_transform()(image.convert("RGB")).unsqueeze(0).to(device)
