import torch
import torch.nn.functional as F

import numpy as np
import cv2

from PIL import Image
from deeptect.video.GenD.model import GenD

def get_target_layer(model):
    fe = model.feature_extractor

# =============================================================================
#     # ---- CLIP ----
# =============================================================================
    if hasattr(fe, "vision_model"):
        return fe.vision_model.encoder.layers[-1]

# =============================================================================
#     # ---- DINO (v2 / v3 HF) ----
# =============================================================================
    if hasattr(fe, "backbone"):
        bb = fe.backbone

# =============================================================================
#         # DINOv3 (HF)
# =============================================================================
        if hasattr(bb, "layer"):
            return bb.layer[-1]

# =============================================================================
#         # DINOv2 (HF)
# =============================================================================
        if hasattr(bb, "encoder") and hasattr(bb.encoder, "layer"):
            return bb.encoder.layer[-1]

# =============================================================================
#         # timm EVA / ViT
# =============================================================================
        if hasattr(bb, "blocks"):
            return bb.blocks[-1]

    raise ValueError("Unsupported backbone for Grad-CAM")

def vit_tokens_to_cam(
    cam_tokens,
    image_size=(224,224),
    patch_size=14
):
    cam_tokens = cam_tokens[1:]  # drop CLS

    num_tokens = cam_tokens.shape[0]
    side = int(np.sqrt(num_tokens))
    num_patches = side * side
    cam_tokens = cam_tokens[:num_patches]
    cam = cam_tokens.reshape(side, side)
    
    # # Handle both single int and tuple for image_size
    # if isinstance(image_size, int):
    # target_size = (image_size, image_size)
    # else:
    #     target_size = image_size  # (width, height)

    cam = F.interpolate(
        cam.unsqueeze(0).unsqueeze(0),
        size=image_size,
        mode="bilinear",
        align_corners=False
    )

    cam = cam.squeeze()
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-6)

    return cam



def overlay_cam_on_image(image, cam, alpha=0.7):
    image_np = np.array(image)
    cam_np = cam.detach().cpu().numpy()



    cam_np = np.uint8(255 * cam_np)
    heatmap = cv2.applyColorMap(cam_np, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = alpha * image_np + (1 - alpha) * heatmap
    overlay = overlay.astype(np.uint8)

    return overlay

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(_, __, output):
            self.activations = output

        def backward_hook(_, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def getcam(self, input_tensor, class_idx=None):
        self.model.zero_grad()

        logits = self.model(input_tensor)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        score = logits[:, class_idx]
        score.backward()

        return self.compute_cam()

    def compute_cam(self):
        weights = self.gradients.mean(dim=1)
        cam = (self.activations[0] * weights.unsqueeze(1)).sum(dim=-1)
        cam = cam.squeeze(0)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-6)

        return cam

if __name__ == "__main__":

    
    image_path = "/data/Kutub/DEEPFAKES/DATASETS/FACES/FF/1_fake/Deepfakes/000_003/000030.jpg"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_name = "yermandy/GenD_DINOv3_L"
    model = GenD.from_pretrained(model_name)
    model.eval().to(device)
    
    image = Image.open(image_path).convert("RGB")
    
    image = image.resize((224,224))
    
    for p in model.feature_extractor.parameters():
        p.requires_grad = True
    
    tensor = model.feature_extractor.preprocess(image)
    tensor = tensor.unsqueeze(0).to(device)
    
    target_layer = get_target_layer(model)
    gradcam = GradCAM(model, target_layer)
    
    cam_tokens = gradcam.getcam(tensor, class_idx=1)
    
    print("CAM tokens shape:", cam_tokens.shape)
    
    cam_map = vit_tokens_to_cam(cam_tokens, image_size=image.size[0])
    
    print("CAM min/max:", cam_map.min().item(), cam_map.max().item())
    
    overlay = overlay_cam_on_image(image, cam_map)
    
    # import matplotlib.pyplot as plt

    # plt.figure(figsize=(6, 6))
    # plt.imshow(overlay)
    # plt.axis("off")
    # plt.title("Grad-CAM")
    # plt.show()



