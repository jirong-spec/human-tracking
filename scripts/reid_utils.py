# reid_utils.py
import torch
import torchreid
import numpy as np
import torchvision.transforms as T

# -----------------------------
# 初始化 ReID 模型
# -----------------------------
reid_model = torchreid.models.build_model(
    name='osnet_x0_25', num_classes=1000, pretrained=True
)
reid_model.to("cuda")
reid_model.eval()

transform = T.Compose([
    T.Resize((256, 128)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_embeddings_batch(patches, device="cuda"):
    tensors = torch.stack([transform(p) for p in patches]).to(device)
    with torch.no_grad():
        feats = reid_model(tensors)
    feats = feats.cpu().numpy()

    feats = feats[:, :128]  # 截取前 128 維
    feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-6)
    return feats



