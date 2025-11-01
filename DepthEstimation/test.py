import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from datasets.nyu_dataset import NYUDataset
from utils import readlines


def tensor_to_numpy_img(t: torch.Tensor) -> np.ndarray:
    """将 PyTorch Tensor 转换为 NumPy 图像格式"""
    if t.ndim == 4:  # [B, 3, H, W]
        t = t[0]  # 取 batch 的第一个样本
    t = t.detach().cpu().clamp(0, 1)  # 确保值在 [0, 1] 范围内
    img = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)  # 转为 [H, W, 3]
    return img


if __name__ == "__main__":
    # 数据路径和文件名列表
    data_path = "/home/jingyang/MonoLDP/DepthEstimation/nyu_data"
    split_file = "/home/jingyang/MonoLDP/DepthEstimation/splits/nyu/train_files.txt"
    filenames = readlines(split_file)

    # 初始化数据集
    dataset = NYUDataset(
        data_path=data_path,
        filenames=filenames,
        height=256,
        width=320,
        frame_idxs=[0, -2, 2],  # 当前帧、前一帧、后一帧
        num_scales=4,
        is_train=True,
        img_ext=".jpg",
    )

    # 初始化 DataLoader
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,  # 便于调试
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    print(f"Dataset length: {len(dataset)}")

    # 遍历 DataLoader
    for i, batch in enumerate(loader):
        print(f"\nSample #{i}")

        # 遍历尺度和帧，显示图片
        frame_ids = dataset.frame_idxs
        num_scales = dataset.num_scales if hasattr(dataset, "num_scales") else 1

        for s in range(num_scales):
            fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(8, 9))
            fig.suptitle(f"Sample #{i} - Scale {s}", fontsize=12)

            # 行：帧（前一帧、当前帧、后一帧）；列：原图(color)、增强图(color_aug)
            for r, fid in enumerate(frame_ids):
                for c, kind in enumerate(["color", "color_aug"]):
                    key = (kind, fid, s)
                    ax = axes[r, c]
                    ax.axis("off")  # 关闭坐标轴
                    if key in batch:
                        img = tensor_to_numpy_img(batch[key])  # 转换为 NumPy 图像
                        ax.imshow(img)  # 显示图像
                        ax.set_title(f"{kind} fid={fid} {img.shape[1]}x{img.shape[0]}")
                    else:
                        ax.set_title(f"{kind} fid={fid} (missing)")

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            print(f"  Showing scale={s} ... 关闭窗口继续")
            plt.show()  # 阻塞，关闭窗口后继续

        # 如需只看前 N 个样本，取消下一行注释
        if i >= 2: break