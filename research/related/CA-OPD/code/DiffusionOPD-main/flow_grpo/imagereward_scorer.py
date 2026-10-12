import os
from PIL import Image
import torch
import ImageReward as RM


class ImageRewardScorer(torch.nn.Module):
    def __init__(self, device="cuda", dtype=torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.model = (
            RM.load(
                "ImageReward-v1.0",
                device=device,
                download_root=os.path.join(os.environ.get("HF_HOME", "~/.cache/"), "ImageReward"),
            )
            .eval()
            .to(dtype=dtype)
        )
        self.model.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, prompts, images):
        _, rewards = self.model.inference_rank(prompts, images)
        # ⚠️ 上游 bug:inference_rank 在**只有 1 个 prompt** 时返回标量 float 而非列表,
        # torch.Tensor(float) 会抛 "data must be a sequence (got float)"。
        # 当 per-rank 样本数 % eval_batch_size == 1 时必然触发(如 3 卡 × 256 样本 -> 86%5=1)。
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).reshape(-1)
        n = len(prompts)
        if rewards.numel() == n * n:          # 全配对矩阵 -> 取对角线
            rewards = torch.diagonal(rewards.reshape(n, n), 0)
        return rewards.contiguous()


# Usage example
def main():
    scorer = ImageRewardScorer(device="cuda", dtype=torch.float32)

    images = [
        "test_cases/nasa.jpg",
        "test_cases/hello world.jpg",
    ]
    pil_images = [Image.open(img) for img in images]
    prompts = [
        'An astronaut’s glove floating in zero-g with "NASA 2049" on the wrist',
        'New York Skyline with "Hello World" written with fireworks on the sky',
    ]
    print(scorer(prompts, pil_images))


if __name__ == "__main__":
    main()
