from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from utils._utils import *
import matplotlib.colors as colors
from scipy import stats



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

def fft_heatmap(data, title="FFT Magnitude Spectrum", log_scale=True,
                cmap="viridis", figsize_per_channel=(4, 4), shift=True,
                normalize=False, save_path=None, clim=None):
    """
    对 [C, H, W] 形状的输入数据，逐通道做 2D FFT，绘制模长热图。

    Parameters
    ----------
    data        : array-like，支持 numpy.ndarray / torch.Tensor / list
                  shape 必须为 [C, H, W]
    title       : 大图总标题
    log_scale   : bool，是否对幅度取 log（推荐 True，否则直流分量会掩盖其他信息）
    cmap        : colormap，默认 "hot"（也可以用 "inferno"、"magma" 等）
    figsize_per_channel : 每个子图的尺寸 (w, h)，单位 inch
    shift       : bool，是否做 fftshift（将零频移到中心，推荐 True）
    normalize   : bool，是否将每个通道的幅度图归一化到 [0, 1]
    save_path   : str or None，若提供则保存图片到该路径

    Returns
    -------
    fig         : matplotlib.figure.Figure
    magnitude   : np.ndarray, shape [C, H, W]，各通道 FFT 幅度（未取 log）
    """

    backgrand_color = "#f5f5f5"
    tex_color = "#000000"
    
    # ── 1. 统一转换为 numpy float32 ──────────────────────────────────────────
    if hasattr(data, "detach"):          # torch.Tensor
        arr = data.detach().cpu().numpy()
    elif not isinstance(data, np.ndarray):
        arr = np.array(data)
    else:
        arr = data

    arr = arr.astype(np.float32)

    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]          # [1, H, W]
    
    if arr.ndim != 3:
        raise ValueError(f"输入维度应为 3 (C, H, W)，实际为 {arr.ndim}")

    C, H, W = arr.shape

    # ── 2. 逐通道 FFT ────────────────────────────────────────────────────────
    magnitude = np.zeros((C, H, W), dtype=np.float32)

    for c in range(C):
        f = np.fft.fft2(arr[c])
        if shift:
            f = np.fft.fftshift(f)
        magnitude[c] = np.abs(f)

    # ── 3. 准备绘图数据 ───────────────────────────────────────────────────────
    display = magnitude.copy()

    if log_scale:
        display = np.log1p(display)          # log(1 + |F|)，避免 log(0)

    if normalize:
        for c in range(C):
            vmin, vmax = display[c].min(), display[c].max()
            if vmax > vmin:
                display[c] = (display[c] - vmin) / (vmax - vmin)

    # ── 4. 排列子图布局 ───────────────────────────────────────────────────────
    ncols = min(C, 8)                        # 每行最多 8 个
    nrows = int(np.ceil(C / ncols))

    fw = figsize_per_channel[0] * ncols + 1
    fh = figsize_per_channel[1] * nrows + 1.2

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(fw, fh),
                             squeeze=False)

    fig.patch.set_facecolor(backgrand_color)
    fig.suptitle(title, fontsize=14, color=tex_color, fontweight="bold", y=1.01)

    for c in range(C):
        r, col = divmod(c, ncols)
        ax = axes[r][col]
        vmin, vmax = clim if clim is not None else (None, None)
        im = ax.imshow(display[c], cmap=cmap, interpolation="nearest",
               aspect="auto", vmin=vmin, vmax=vmax)

        ax.set_title(f"Ch {c}", fontsize=9, color=tex_color, pad=3)
        ax.axis("off")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=6, colors=tex_color)
        cbar.outline.set_edgecolor("#333333")

        ax.set_facecolor(backgrand_color)

    # 隐藏多余的子图格
    for idx in range(C, nrows * ncols):
        r, col = divmod(idx, ncols)
        axes[r][col].set_visible(False)

    label = ("log(1+|FFT|)" if log_scale else "|FFT|") + \
            (" [normalized]" if normalize else "")
    fig.text(0.5, -0.01, label, ha="center", fontsize=9,
             color="#888888", style="italic")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"图片已保存至: {save_path}")

    plt.show()
    return fig, magnitude

def visualize_results(gt, pred, idx, s, title="Model Prediction Analysis", save_path = "./test_pic"):
    # 1. 计算残差
    psnr = PSNR()
    ssim = SSIM()
    
    psnr_value = psnr(pred, gt)
    ssim_value = ssim(pred, gt)
    
    pred = pred.permute(1, 2, 0).numpy()
    gt = gt.permute(1, 2, 0).numpy()
    res = gt - pred
    # 2. 设置画布：1行3列
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    
    # 定义显示参数，确保对比公平
    # 如果数据是 0-1 归一化的，我们可以固定 vmin/vmax
    display_params = {'cmap': 'gray', 'vmin': gt.min(), 'vmax': gt.max()}

    # --- 第一张图：Ground Truth ---
    im0 = axes[0].imshow(gt, **display_params)
    axes[0].set_title("Ground Truth")
    axes[0].axis('off')
    
    # --- 第二张图：Prediction ---
    im1 = axes[1].imshow(pred, **display_params)
    axes[1].set_title(f"Prediction\nPSNR: {psnr_value:.2f} dB SSIM: {ssim_value:.4f}")
    axes[1].axis('off')
    
    # --- 第三张图：Residual (加上 Colorbar) ---
    # 使用 'seismic' 或 'jet' 映射，这样误差大小一目了然
    # vmin/vmax 建议根据残差的实际分布设置，或者取对称范围
    res_abs = np.abs(res)
    max_err = np.max(np.abs(res))
    min_err = np.min(np.abs(res))
    print(f"Max Residual: {max_err:.4f}")
    print(f"Min Residual: {min_err:.4f}")
    print(f"Mean Residual: {np.mean(res_abs):.4f}")
    print(f"Median Residual: {np.median(res_abs):.4f}")
    # visualize_distribution(res_abs)
    res_flattened = res_abs.flatten()
    p95 = np.percentile(res_flattened, 95)
    res_cut = np.zeros_like(res_abs)
    res_cut[res_abs > p95] = 1  # 标记超过95百分位的残差
    print(f"95th Percentile Residual: {p95:.4f}")
    res_mean_channel = np.mean(res_abs, axis=2)  # 计算每个像素的平均残差
    # p_low  = np.percentile(res_abs, 2)
    # p_high = np.percentile(res_abs, 98)
    # clipped = np.clip(res_abs, p_low, p_high)
    # log_res_norm = (clipped - p_low) / (p_high - p_low + 1e-8)

    # im2 = axes[2].imshow(log_res_norm, cmap='hot', vmin=0, vmax=1)
    # log_res = np.log(res_abs + 1e-8)  # log(1 + |res|)
    # print(f"Max Log Residual: {log_res.max():.4f}")
    # print(f"Min Log Residual: {log_res.min():.4f}")
    # print(f"Mean Log Residual: {log_res.mean():.4f}")
    # print(f"Median Log Residual: {np.median(log_res):.4f}")
    # alpha = 100  # 调整参数，控制压缩程度
    # res_log = np.log1p(alpha * res_abs) / np.log1p(alpha * res_abs.max())
    # im2 = axes[2].imshow(res_log, cmap='magma')
    
    log_norm = colors.LogNorm(vmin=1e-4, vmax=res_mean_channel.max())
    im2 = axes[2].imshow(res_mean_channel, cmap='magma', norm=log_norm)
    # im2 = axes[2].imshow(res_abs, **display_params) 

    axes[2].set_title("Residual (GT - Pred)")
    axes[2].axis('off')

    # 在右侧添加 Colorbar (Scale)
    # fraction 和 pad 参数用于调整 colorbar 的大小和间距
    cbar = fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label('Error Value')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    return fig, axes
    plt.savefig(
            os.path.join(save_path, f'x{s}_{idx:04d}.png'), 
            dpi=600,                # 600 DPI 是出版级高清标准，300 是基本要求
            bbox_inches='tight',    # 自动裁剪多余的白边
            pad_inches=0.1,         # 边缘留白的宽度
            transparent=False)      # 如果需要透明背景可设为 True
    plt.show()



def fft_energy_analysis(arr, plot=True, fig_title=None, save_path = None):
    """
    对输入的 tensor/array 进行 FFT 频谱能量分析，并绘制中心矩形区域能量占比曲线。

    参数
    ----
    arr   : array-like，shape 为 [H, W] 或 [C, H, W]
    plot  : bool，是否绘图，默认 True
    fig_title : str，图标题（可选）

    返回
    ----
    x     : np.ndarray，linspace(0, 1, 101)
    y     : np.ndarray，各尺度矩形内能量 / 全图能量
    fft_magnitude : np.ndarray，FFT 幅值谱（已移中心），shape [C, H+1, W+1] 或 [H+1, W+1]
    """
    arr = np.asarray(arr, dtype=np.float64)

    # ---------- 统一为 [C, H, W] ----------
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]          # [1, H, W]
        squeeze_output = True
    elif arr.ndim == 3:
        squeeze_output = False
    else:
        raise ValueError(f"输入 shape 应为 [H,W] 或 [C,H,W]，得到 {arr.shape}")

    C, H, W = arr.shape

    # ---------- FFT，zero-pad 到 (H+1, W+1) 使 0 频严格居中 ----------
    # np.fft.fft2 支持 s 参数指定输出尺寸
    new_H, new_W = H + 1, W + 1
    fft_shifted_list = []

    for c in range(C):
        F = np.fft.fft2(arr[c], s=(new_H, new_W))   # FFT 并补零
        F_shifted = np.fft.fftshift(F)               # 0 频移至中心
        fft_shifted_list.append(F_shifted)

    fft_shifted = np.stack(fft_shifted_list, axis=0)          # [C, H+1, W+1]
    power = np.abs(fft_shifted) ** 2                           # 功率谱

    # ---------- 坐标系：以中心为 (0,0) ----------
    cy, cx = new_H // 2, new_W // 2     # 中心像素坐标（整数）
    # 对应频率轴索引（相对于中心）
    iy = np.arange(new_H) - cy          # [-cy, ..., 0, ..., new_H-1-cy]
    ix = np.arange(new_W) - cx

    # ---------- x & y 计算 ----------
    x = np.linspace(0, 1, 101)

    # 每个 channel 各自的全图能量
    total_energy_per_channel = power.sum(axis=(1, 2))   # [C]

    # 对每个 x，统计矩形 [-H//2*x, H//2*x] × [-W//2*x, W//2*x] 内的能量
    half_H = H // 2
    half_W = W // 2

    y = np.zeros_like(x)
    for i, xi in enumerate(x):
        # 矩形边界（频率索引）
        ry = half_H * xi    # 行方向半径
        rx = half_W * xi    # 列方向半径

        # 布尔掩码：哪些像素落在矩形内
        mask_y = np.abs(iy) <= ry    # [H+1]
        mask_x = np.abs(ix) <= rx    # [W+1]
        mask = np.outer(mask_y, mask_x)   # [H+1, W+1]

        # 对所有 channel 求和后再除以各自总能量，取均值
        rect_energy = (power * mask[np.newaxis, :, :]).sum(axis=(1, 2))   # [C]
        # 避免除以 0
        ratio = np.where(
            total_energy_per_channel > 0,
            rect_energy / total_energy_per_channel,
            0.0
        )
        y[i] = ratio.mean()

    # ---------- 输出形状处理 ----------
    if squeeze_output:
        fft_magnitude = np.abs(fft_shifted[0])   # [H+1, W+1]
    else:
        fft_magnitude = np.abs(fft_shifted)       # [C, H+1, W+1]

    # ---------- 绘图 ----------
    if plot:
        _plot_results(x, y, power, squeeze_output, fft_shifted, half_H, half_W,
                      cy, cx, new_H, new_W, C, fig_title, save_path)

    return x, y, fft_magnitude


# ─────────────────────────────────────────────
def _plot_results(x, y, power, squeeze_output, fft_shifted,
                half_H, half_W, cy, cx, new_H, new_W, C, fig_title, save_path):
    """内部绘图函数"""

    # 用 log-scale 幅值谱做可视化（取 channel 均值）
    log_mag = np.log1p(np.abs(fft_shifted).mean(axis=0))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    if fig_title:
        fig.suptitle(fig_title, fontsize=14, fontweight='bold')

    # ── 左图：FFT 幅值谱 + 几个示例矩形 ──────────────────────────────
    ax0 = axes[0]
    im = ax0.imshow(log_mag, cmap='inferno', origin='upper',
                    extent=[-cx, new_W - 1 - cx, cy, -(new_H - 1 - cy)])
    plt.colorbar(im, ax=ax0, fraction=0.046, pad=0.04, label='log(1 + |F|)')

    ax0.axhline(0, color='white', linewidth=0.5, alpha=0.4)
    ax0.axvline(0, color='white', linewidth=0.5, alpha=0.4)

    # 叠加几个示例矩形（x = 0.25, 0.5, 0.75, 1.0）
    colors = ['#00e5ff', '#69ff47', '#ff6d00', '#ff1744']
    for xi, col in zip([0.25, 0.5, 0.75, 1.0], colors):
        rh = half_H * xi
        rw = half_W * xi
        rect = Rectangle((-rw, -rh), 2 * rw, 2 * rh,
                        linewidth=1.4, edgecolor=col,
                        facecolor='none', linestyle='--',
                        label=f'x={xi:.2f}')
        ax0.add_patch(rect)

    ax0.set_title('FFT Energy', fontsize=12)
    ax0.set_xlabel('W')
    ax0.set_ylabel('H')
    ax0.legend(fontsize=8, loc='upper right',
            facecolor='#1a1a2e', labelcolor='white')

    # ── 右图：x vs (1-y) log scale ──────────────────────────────────
    ax1 = axes[1]

    eps = 1e-9
    one_minus_y = np.clip(1.0 - y, eps, None)   # 缺失能量比例，防止 log(0)

    ax1.fill_between(x, one_minus_y, eps, alpha=0.18, color='#4fc3f7')
    ax1.plot(x, one_minus_y, color='#0288d1', linewidth=2.2, label='1 - y  (missing energy ratio)')
    ax1.set_yscale('log')

    # 标注几个关键点
    for xi, col in zip([0.25, 0.5, 0.75, 1.0], colors):
        idx = int(round(xi * 100))
        yv = one_minus_y[idx]
        ax1.scatter(x[idx], yv, color=col, zorder=5, s=60)
        ax1.annotate(f'x={xi:.2f}\n1-y={yv:.2e}',
                    xy=(x[idx], yv),
                    xytext=(x[idx] + 0.04, yv * 3),
                    fontsize=7.5, color=col,
                    arrowprops=dict(arrowstyle='->', color=col, lw=1))

    # 右侧次坐标轴：显示原始 y 值（线性）
    ax1r = ax1.twinx()
    ax1r.plot(x, y, color='#ff7043', linewidth=1.2,
            linestyle=':', alpha=0.7, label='y  (energy ratio)')
    ax1r.set_ylabel('y  (Energy Ratio, linear)', fontsize=10, color='#ff7043')
    ax1r.tick_params(axis='y', labelcolor='#ff7043')
    ax1r.set_ylim(0, 1.05)

    ax1.set_xlim(-0.01, 1.02)
    ax1.set_xlabel('x  (Rectangle Scale)', fontsize=11)
    ax1.set_ylabel('1 - y  (Missing Energy, log scale)', fontsize=10)
    ax1.set_title('Central Rectangle Energy Proportion Curve', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.4, which='both')

    # 合并两个轴的图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1r.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='lower left')

    plt.tight_layout()
    if save_path:
        plt.savefig('/mnt/user-data/outputs/fft_energy_curve.png',
                    dpi=150, bbox_inches='tight')
        print("图像已保存至 fft_energy_curve.png")
    plt.show()
    

# ─────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────
if __name__ == '__main__':
    rng = np.random.default_rng(42)

    # 示例 1：单通道 [H, W] —— 含低频结构的图像
    H, W = 64, 64
    xx, yy = np.meshgrid(np.linspace(0, 4 * np.pi, W),
                         np.linspace(0, 4 * np.pi, H))
    img_2d = np.sin(xx) * np.cos(yy) + 0.3 * rng.standard_normal((H, W))

    print("=== 单通道 [H,W] 测试 ===")
    x, y, fft_mag = fft_energy_analysis(img_2d, fig_title='单通道低频图像')
    print(f"fft_magnitude shape: {fft_mag.shape}")   # (H+1, W+1)
    print(f"y(x=0.5) = {y[50]:.4f}")

    # 示例 2：三通道 [C, H, W]
    img_3c = rng.standard_normal((3, H, W))
    img_3c[0] += np.sin(xx) * 2   # 第 0 通道含低频分量

    print("\n=== 三通道 [C,H,W] 测试 ===")
    x2, y2, fft_mag2 = fft_energy_analysis(img_3c, fig_title='三通道混合图像')
    print(f"fft_magnitude shape: {fft_mag2.shape}")  # (3, H+1, W+1)