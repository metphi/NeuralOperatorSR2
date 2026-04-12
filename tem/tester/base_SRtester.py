from abc import ABC, abstractmethod
from utils._utils import PSNR, SSIM
import matplotlib.colors as colors
from scipy import stats
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle







class BaseSRTester(ABC):
    def __init__(self, model, dataset, device = 'cpu'):
        self.model = model
        self.dataset = dataset
        self.ssim = SSIM()
        self.psnr = PSNR()
        print(f"Model has {self._count_parameters(model)} trainable parameters.")
        

    def _count_parameters(model):
        """count model parameters"""
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params
    
    
    
    
      
    def test(self, idx, patch_h, patch_w):
        if type(idx) == int:
            pass
        if type(idx) == list:
            pass
        if type == 'all':
            pass     
        
        
        
        
        
    def visualize_distribution(arr):
        data = arr.flatten()

        fig, ax = plt.subplots(figsize=(10, 5))

        # 直方图（概率密度归一化）
        ax.hist(data, bins='auto', density=True, alpha=0.4, color='steelblue', label='Histogram')

        # KDE 曲线（大数组用 bw_method 控制带宽，默认 Scott）
        kde = stats.gaussian_kde(data, bw_method='scott')
        x = np.linspace(data.min(), data.max(), 1000)
        ax.plot(x, kde(x), color='limegreen', linewidth=2, label='KDE')

        # 均值 / 中位数
        ax.axvline(data.mean(),   color='tomato',     linestyle='--', linewidth=1.5, label=f'Mean={data.mean():.4g}')
        ax.axvline(np.median(data), color='gold',     linestyle='--', linewidth=1.5, label=f'Median={np.median(data):.4g}')

        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.set_title('Probability Density Distribution')
        ax.legend()
        plt.tight_layout()
        plt.show()

        # 打印基本统计量
        data_rms = torch.sqrt(torch.mean(data**2))
        print(f"Shape : {arr.shape}  →  flattened N={len(data):,}")
        print(f"Mean  : {data.mean():.6g}")
        print(f"Std   : {data.std():.6g}")
        print(f"std/rms:{data.std()/data_rms:.6g}")
        print(f"Min   : {data.min():.6g}   Max: {data.max():.6g}")
        print(f"Q1/Q3 : {np.percentile(data,25):.6g} / {np.percentile(data,75):.6g}")
        
    
    