# Based on https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/fe88a163f4661b4ddabba0751ff645e2e620746e/simple_inference.py

import os
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from flow_grpo.reward_ckpt_path import CKPT_PATH
import numpy as np
from PIL import Image


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    @torch.no_grad()
    def forward(self, embed):
        return self.layers(embed)


class AestheticScorer(torch.nn.Module):
    def __init__(self, dtype, device):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.mlp = MLP().to(device)
        state_dict = torch.load(os.path.join(CKPT_PATH, "sac+logos+ava1-l14-linearMSE.pth"), map_location="cpu")
        self.mlp.load_state_dict(state_dict)
        self.dtype = dtype
        self.device = device
        self.eval()

    @torch.no_grad()
    def __call__(self, images):
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.dtype).to(self.device) for k, v in inputs.items()}
        embed = self.clip.get_image_features(**inputs)
        # normalize embedding
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return self.mlp(embed).squeeze(1)


# Usage example
def main():
    scorer = AestheticScorer(device="cuda", dtype=torch.float32)

    images = [
        "test_cases/nasa.jpg",
    ]
    pil_images = np.stack([np.array(Image.open(img)) for img in images])
    images = pil_images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
    images = torch.tensor(images, dtype=torch.uint8)
    print(scorer(images))


if __name__ == "__main__":
    main()


class DiffAestheticScorer(torch.nn.Module):
    """可微版 AestheticScorer:直接吃 [B,3,H,W] in [0,1] 的 tensor,自己做 CLIP 预处理,
    保留计算图 —— 用于 IT-OPD 的 reward 梯度,以及"真 teacher 方向 vs 局部 reward 梯度方向"
    的**同 reward 对照**(原 AestheticScorer 走 PIL processor 且 @no_grad,无法反传)。
    接口对齐 PickScoreScorer 以便互换。
    """

    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(self, device="cuda", dtype=torch.float32):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
        self.mlp = MLP().to(device).eval()
        self.mlp.load_state_dict(
            torch.load(os.path.join(CKPT_PATH, "sac+logos+ava1-l14-linearMSE.pth"), map_location="cpu")
        )
        for _p in self.parameters():
            _p.requires_grad_(False)
        self.device, self.dtype = device, dtype
        self.register_buffer("_m", torch.tensor(self.CLIP_MEAN, device=device).view(1, 3, 1, 1))
        self.register_buffer("_s", torch.tensor(self.CLIP_STD, device=device).view(1, 3, 1, 1))

    def encode_text(self, prompts):
        # aesthetic 与 prompt 无关,但调用方会把返回值存进 sample dict 再 torch.cat,
        # 返回 None 会导致 "expected Tensor ... but got NoneType" -> 返回占位 tensor。
        return torch.zeros(len(prompts), 1, device=self.device, dtype=self.dtype)

    def score_from_images(self, text_embs, images01):
        x = torch.nn.functional.interpolate(
            images01, size=(224, 224), mode="bicubic", align_corners=False, antialias=True
        )
        x = (x - self._m) / self._s
        e = self.clip.get_image_features(pixel_values=x.to(self.dtype))
        e = e / torch.linalg.vector_norm(e, dim=-1, keepdim=True)
        # MLP.forward 带 @no_grad -> 直接走 layers 以保留计算图
        return self.mlp.layers(e).squeeze(1)
