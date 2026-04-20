import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchkbnufft as tkbn


def count_parameters(model):
    """count model parameters"""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params


def rgb_to_y(img: torch.Tensor, standard: str = "ITU-R BT.601") -> torch.Tensor:
    """
    RGB -> Y (亮度通道), ITU-R BT.601
    img: (B, 3, H, W) or (3, H, W), range [0, 1]
    return: (B, 1, H, W) or (1, H, W), range [16/255, 235/255]
    """
    if img.ndim == 3:
        img = img.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    y = 16.0/255.0 + (65.481/255.0)*r + (128.553/255.0)*g + (24.966/255.0)*b

    return y.squeeze(0) if squeeze else y


class PSNR(nn.Module):
    """
    标准SR评估PSNR:
    - 只在Y通道计算
    - 裁掉scale个像素的边缘
    """
    def __init__(self, scale: int = 2):
        super().__init__()
        self.scale = scale

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        """
        sr, hr: (B, 3, H, W) or (3, H, W), range [0, 1]
        return: scalar tensor, batch内各图PSNR的算术平均
        """
        if sr.ndim == 3:
            sr = sr.unsqueeze(0)
            hr = hr.unsqueeze(0)

        # 1. 转Y通道
        sr_y = rgb_to_y(sr)  # (B, 1, H, W)
        hr_y = rgb_to_y(hr)

        # 2. 裁边
        s = self.scale
        sr_y = sr_y[..., s:-s, s:-s]
        hr_y = hr_y[..., s:-s, s:-s]

        # 3. 每张图单独算MSE，再各自转PSNR，最后取平均
        #    注意：不能先平均MSE再取log，对数是非线性的！
        mse = torch.mean((sr_y - hr_y) ** 2, dim=[1, 2, 3])  # (B,)
        mse = mse.clamp(min=1e-10)
        psnr = 10.0 * torch.log10(1.0 / mse)                 # max_val=1.0

        return psnr.mean()


class SSIM(nn.Module):
    """
    标准SR评估SSIM:
    - 只在Y通道计算
    - 裁掉scale个像素的边缘
    """
    def __init__(self, scale: int = 2, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.scale       = scale
        self.window_size = window_size
        # Y通道 -> 1个channel
        self.register_buffer('window', self._create_window(window_size, sigma))

    def _create_window(self, size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g      = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g     /= g.sum()
        g_2d   = g.unsqueeze(1) @ g.unsqueeze(0)       # (size, size)
        return g_2d.unsqueeze(0).unsqueeze(0)           # (1, 1, size, size)

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        if sr.ndim == 3:
            sr = sr.unsqueeze(0)
            hr = hr.unsqueeze(0)

        # 1. 转Y通道
        sr_y = rgb_to_y(sr)  # (B, 1, H, W)
        hr_y = rgb_to_y(hr)

        # 2. 裁边
        s    = self.scale
        sr_y = sr_y[..., s:-s, s:-s]
        hr_y = hr_y[..., s:-s, s:-s]

        # 3. 计算SSIM（单通道，groups=1）
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        pad = self.window_size // 2

        mu1    = F.conv2d(sr_y, self.window, padding=pad)
        mu2    = F.conv2d(hr_y, self.window, padding=pad)
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1mu2 = mu1 * mu2

        s1     = F.conv2d(sr_y * sr_y, self.window, padding=pad) - mu1_sq
        s2     = F.conv2d(hr_y * hr_y, self.window, padding=pad) - mu2_sq
        s12    = F.conv2d(sr_y * hr_y, self.window, padding=pad) - mu1mu2

        num    = (2 * mu1mu2 + c1) * (2 * s12    + c2)
        den    = (mu1_sq + mu2_sq + c1) * (s1 + s2 + c2)

        return (num / den).mean()
    

def tile_inference(
    lr_tensor,
    model,
    scale=2,
    only_head=False,
    tile=128,
    halo=16,
    norm={'lr': {'sub': 0., 'div': 1.},
          'hr': {'sub': 0., 'div': 1.}},
):
    """
    Tile-based inference for arbitrary-scale SR models (SRNO / GSASR style).

    Strategy:
      - Reflect-pad the whole LR by `halo` on each side, so every tile can
        read its halo without boundary checks.
      - Slide non-overlapping CORE regions of size `tile` across the LR.
        The last core region along each axis snaps to the image edge
        (may overlap with the previous core; later writes overwrite).
      - For each core region, crop LR tile = core + halo (from padded LR),
        run the model, crop out the corresponding HR core region from the
        model's output, and paste into the HR canvas.

    Args:
        lr_tensor : (B, C, H, W) in the original value range.
        model     : callable, takes LR tile of shape (B, C, h, w) and returns
                    either an HR tensor of shape (B, C, round(h*scale), round(w*scale))
                    or a tuple (sr, res) following your previous convention.
        scale     : float or int, upsampling factor (supports non-integer).
        only_head : if model returns (sr, res), whether to use `sr` alone
                    (True) or `sr + 0.5 * res` (False).
        tile      : LR-side core tile size (pixels). Must be > 0.
        halo      : LR-side halo padding per side (pixels). Must be >= 0.
        norm      : dict with 'lr' and 'hr' each having 'sub' and 'div'.
                    Input is normalized as (lr - lr_sub) / lr_div,
                    output is de-normalized as sr * hr_div + hr_sub.

    Returns:
        sr_tensor : (B, C, round(H*scale), round(W*scale)) in the original value range.
    """
    assert lr_tensor.dim() == 4, "lr_tensor must be (B, C, H, W)"
    assert tile > 0 and halo >= 0
    B, C, H, W = lr_tensor.shape
    device = lr_tensor.device
    dtype  = lr_tensor.dtype

    # Target HR size (supports non-integer scale)
    H_hr = int(round(H * scale))
    W_hr = int(round(W * scale))

    # --- 1. Normalize input ---
    lr_sub = float(norm['lr']['sub'])
    lr_div = float(norm['lr']['div'])
    hr_sub = float(norm['hr']['sub'])
    hr_div = float(norm['hr']['div'])
    lr_n = (lr_tensor - lr_sub) / lr_div

    # --- 2. Reflect-pad LR by `halo` on each side ---
    # After padding, for any core region [y0:y0+tile] in the ORIGINAL LR,
    # we read the padded LR at [y0 : y0 + tile + 2*halo] (shifted by +halo).
    if halo > 0:
        lr_pad = F.pad(lr_n, (halo, halo, halo, halo), mode='reflect')
    else:
        lr_pad = lr_n

    # --- 3. Compute core-region start coordinates (last one snaps to edge) ---
    def _core_starts(length, core):
        if length <= core:
            return [0]
        starts = list(range(0, length - core + 1, core))
        if starts[-1] != length - core:
            starts.append(length - core)   # snap last core to the right/bottom edge
        return starts

    ys = _core_starts(H, tile)
    xs = _core_starts(W, tile)

    # --- 4. Output canvas ---
    sr_tensor = torch.zeros(B, C, H_hr, W_hr, device=device, dtype=dtype)

    # --- 5. Slide over cores ---
    with torch.no_grad():
        for y0 in ys:
            # core height in LR (handles images smaller than `tile`)
            core_h = min(tile, H - y0)
            y1 = y0 + core_h

            # HR-side core coordinates (integer-rounded, contiguous)
            y0_hr = int(round(y0 * scale))
            y1_hr = int(round(y1 * scale))

            for x0 in xs:
                core_w = min(tile, W - x0)
                x1 = x0 + core_w

                x0_hr = int(round(x0 * scale))
                x1_hr = int(round(x1 * scale))

                # --- 5a. Crop LR tile (core + halo) from the padded LR ---
                # In padded coords, the core sits at [y0+halo : y1+halo],
                # so the tile-with-halo is [y0 : y1+2*halo].
                lr_tile = lr_pad[:, :,
                                 y0 : y1 + 2 * halo,
                                 x0 : x1 + 2 * halo]

                # --- 5b. Forward pass ---
                out = model(lr_tile)
                if isinstance(out, tuple):
                    sr_tile, res_tile = out[0], out[1]
                    if not only_head:
                        sr_tile = sr_tile + 0.5 * res_tile
                else:
                    sr_tile = out

                # --- 5c. Compute where to crop the core region out of sr_tile ---
                # The LR tile spans y0_lr_eff = y0 - halo (in original LR coords,
                # may be negative — that's fine, it's handled by reflect padding).
                # Its HR projection starts at round((y0 - halo) * scale).
                # The core in HR starts at y0_hr, so the offset inside sr_tile is:
                lr_tile_y0_eff = y0 - halo
                lr_tile_x0_eff = x0 - halo
                crop_y0 = y0_hr - int(round(lr_tile_y0_eff * scale))
                crop_x0 = x0_hr - int(round(lr_tile_x0_eff * scale))
                crop_y1 = crop_y0 + (y1_hr - y0_hr)
                crop_x1 = crop_x0 + (x1_hr - x0_hr)

                core_hr = sr_tile[:, :, crop_y0:crop_y1, crop_x0:crop_x1]

                # Sanity check: shape must match the HR core slot exactly.
                expected_h = y1_hr - y0_hr
                expected_w = x1_hr - x0_hr
                if core_hr.shape[-2:] != (expected_h, expected_w):
                    # Fallback: if the model's output is off by 1-2 px due to
                    # internal rounding, resize to fit. Rare with well-behaved models.
                    core_hr = F.interpolate(
                        core_hr, size=(expected_h, expected_w),
                        mode='bilinear', align_corners=False,
                    )

                # --- 5d. Paste into HR canvas (later writes overwrite earlier) ---
                sr_tensor[:, :, y0_hr:y1_hr, x0_hr:x1_hr] = core_hr

    # --- 6. De-normalize output ---
    sr_tensor = sr_tensor * hr_div + hr_sub
    return sr_tensor

    
# def patch_inference(model, lr_tensor, patch_size=128, overlap=16, scale=2, only_head = False):
#     """
#     lr_tensor: (1, 3, H, W), range [0,1]
#     """
#     B, C, H, W = lr_tensor.shape
#     stride = patch_size - overlap
    
#     sr_tensor = torch.zeros(B, C, H*scale, W*scale, device=lr_tensor.device)
#     count_map = torch.zeros_like(sr_tensor)

#     for top in range(0, H, stride):
#         for left in range(0, W, stride):
#             # 边缘 clamp，保证 patch 大小始终是 patch_size
#             t = min(top,  H - patch_size)
#             l = min(left, W - patch_size)

#             patch = lr_tensor[:, :, t:t+patch_size, l:l+patch_size]

#             with torch.no_grad():
#                 sr_patch = model(patch)
#                 if isinstance(sr_patch, tuple):
#                     sr_patch, res_patch = sr_patch[0], sr_patch[1]
#                     if only_head:
#                         pass
#                     else:
#                         sr_patch = sr_patch + 0.5 * res_patch

#             # overlap 区域多次累加，最终取平均（软融合）
#             sr_tensor[:, :,
#                       t*scale:(t+patch_size)*scale,
#                       l*scale:(l+patch_size)*scale] += sr_patch
#             count_map[:, :,
#                       t*scale:(t+patch_size)*scale,
#                       l*scale:(l+patch_size)*scale] += 1

#     return sr_tensor / count_map.clamp(min=1)


def direct_inference(model, lr_tensor, sheet, only_head = False, norm={'lr': {'sub': 0., 'div': 1.}, 'hr': {'sub': 0., 'div': 1.}}):
    """
    直接推理指定的一个sheet。
    """
    with torch.no_grad():
        lr = lr_tensor[sheet]
        lr = (lr - norm['lr']['sub']) / norm['lr']['div']
        sr_tensor = model(lr.unsqueeze(0))  # [1, C, H*scale, W*scale]
        if isinstance(sr_tensor, tuple):
            sr_tensor, res_tensor = sr_tensor[0], sr_tensor[1]
            if only_head:
                pass
            else:
                sr_tensor = sr_tensor + res_tensor
        sr_tensor = sr_tensor * norm['hr']['div'] + norm['hr']['sub']
    return sr_tensor.squeeze(0)  # [C, H*scale, W*scale]






# def _make_coords(N, k_range, device):
#     """内部辅助：生成 (omega_h, omega_v) 给定 k 范围张量。"""
#     l = torch.arange(-N, N + 1, dtype=torch.float32, device=device)
#     L, K = torch.meshgrid(l, k_range, indexing="ij")   # (2N+1, len_k)
 
#     c1 = 2 * torch.pi / N
#     c2 = 2 * torch.pi / (N * N)
 
#     omega_h = torch.stack([
#         (c1 * K).reshape(-1),
#         (c2 * L * K).reshape(-1),
#     ], dim=0).clamp(-torch.pi, torch.pi)
 
#     omega_v = torch.stack([
#         (c2 * L * K).reshape(-1),
#         (c1 * K).reshape(-1),
#     ], dim=0).clamp(-torch.pi, torch.pi)
 
#     return omega_h, omega_v
 
 
# def build_ppft_coords_full(N: int, device=torch.device("cpu")):
#     """完整坐标，k ∈ [-N/2, N/2]，用于复数输入。"""
#     k = torch.arange(-N // 2, N // 2 + 1, dtype=torch.float32, device=device)
#     return _make_coords(N, k, device)
 
 
# def build_ppft_coords_half(N: int, device=torch.device("cpu")):
#     """半轴坐标，k ∈ [0, N/2]，用于实数输入加速。"""
#     k = torch.arange(0, N // 2 + 1, dtype=torch.float32, device=device)
#     return _make_coords(N, k, device)
 
 
# # ──────────────────────────────────────────────────────────────
# # 2. 共轭填充（纯 torch，可微分）
# # ──────────────────────────────────────────────────────────────
 
# def fill_conjugate_half(F_half: torch.Tensor, N: int) -> torch.Tensor:
#     """
#     将半轴计算结果 (..., 2N+1, N/2+1) 扩展为完整频谱 (..., 2N+1, N+1)。
 
#     F_half 的列 s ∈ [0, N/2] 对应 k = s（非负部分，含 k=0）。
#     完整输出的列 j ∈ [0, N]   对应 k = j - N/2。
 
#     对称性说明:
#       水平扇区每行 i（对应斜率 l）是一条穿过原点的直线，
#       对实数输入，同行内 k 轴满足共轭对称：
#           F[i, k] = conj(F[i, -k])
#       因此负半轴（k = -N/2..-1）只需对同行正半轴取共轭再翻转，
#       l 轴完全不参与。
 
#     填充规则（对 k<0 部分，即 j = 0..N/2-1）:
#         F_full[..., i, j] = conj( F_half[..., i, N/2-j] )
#                                              ↑ 同行，k 轴对应正值
#     """
#     # 正半轴（k=0..N/2）直接作为输出右半部分
#     F_pos = F_half                           # (..., 2N+1, N/2+1)
 
#     # 负半轴（k=-N/2..-1）：
#     #   取同行的 k=1..N/2（去掉 k=0 列），取共轭，再沿 k 轴反序
#     #   -> 结果对应 k=-N/2..-1（从小到大排列）
#     F_neg = torch.conj(F_half[..., 1:])      # (..., 2N+1, N/2)，k=1..N/2 的共轭
#     F_neg = torch.flip(F_neg, dims=[-1])     # k 轴反序 -> k=-N/2..-1
 
#     # 拼接 [k=-N/2..-1 | k=0..N/2] -> (..., 2N+1, N+1)
#     return torch.cat([F_neg, F_pos], dim=-1)
 
 
# # ──────────────────────────────────────────────────────────────
# # 3. PPFT 主函数
# # ──────────────────────────────────────────────────────────────
 
# def ppft2(image: torch.Tensor, norm: str = "ortho"):
#     """
#     2D 伪极坐标傅里叶变换（完全可微分，支持 .backward()）
 
#     参数:
#         image : Tensor，形状 (..., N, N)，N 为偶数
#                 · 实数输入：自动启用共轭对称加速，NUFFT 计算量减半
#                 · 复数输入：计算完整频率点
#         norm  : 'ortho' | 'forward' | 'backward'（同 torch.fft 约定）
 
#     返回:
#         F_h   : 水平扇区，形状 (..., 2N+1, N+1)，复数
#         F_v   : 垂直扇区，形状 (..., 2N+1, N+1)，复数
#     """
#     orig_shape = image.shape
#     N = orig_shape[-1]
#     assert orig_shape[-2] == N, f"输入最后两维必须相同，但得到 {orig_shape[-2]}×{N}"
#     assert N % 2 == 0,          f"N 必须为偶数，但得到 N={N}"
 
#     is_real = not image.is_complex()
#     device  = image.device
#     half    = N // 2
 
#     # 展平 batch 维 -> (B', N, N)，转复数
#     image_2d   = image.reshape(-1, N, N).to(torch.complex64)
#     B_prime    = image_2d.shape[0]
#     image_tkbn = image_2d.unsqueeze(1)    # (B', 1, N, N)
 
#     nufft_ob = tkbn.KbNufft(im_size=(N, N)).to(device)
 
#     # ── 计算频率坐标（无需梯度）─────────────────────────────
#     with torch.no_grad():
#         if is_real:
#             omega_h, omega_v = build_ppft_coords_half(N, device=device)
#             k_cols = half + 1          # 实际计算的列数
#         else:
#             omega_h, omega_v = build_ppft_coords_full(N, device=device)
#             k_cols = N + 1
 
#     # ── NUFFT 前向 ────────────────────────────────────────────
#     out_h = nufft_ob(image_tkbn, omega_h).squeeze(1)   # (B', (2N+1)*k_cols)
#     out_v = nufft_ob(image_tkbn, omega_v).squeeze(1)
 
#     # ── 归一化 ────────────────────────────────────────────────
#     scale = {
#         "ortho":    1.0 / N,
#         "forward":  1.0 / (N * N),
#         "backward": 1.0,
#     }[norm]
#     out_h = out_h * scale
#     out_v = out_v * scale
 
#     # ── reshape 到 (B', 2N+1, k_cols) ────────────────────────
#     out_h = out_h.reshape(B_prime, 2 * N + 1, k_cols)
#     out_v = out_v.reshape(B_prime, 2 * N + 1, k_cols)
 
#     # ── 实数输入：共轭对称填充完整频谱 ───────────────────────
#     if is_real:
#         out_h = fill_conjugate_half(out_h, N)   # (B', 2N+1, N+1)
#         out_v = fill_conjugate_half(out_v, N)
 
#     # ── 恢复原始 batch 形状 ───────────────────────────────────
#     out_shape = orig_shape[:-2] + (2 * N + 1, N + 1)
#     return out_h.reshape(out_shape), out_v.reshape(out_shape)
 
 
# def ppft2_stacked(image: torch.Tensor, norm: str = "ortho"):
#     """
#     返回: F，形状 (..., 2, 2N+1, N+1)
#           F[..., 0, :, :] = 水平扇区
#           F[..., 1, :, :] = 垂直扇区
#     """
#     F_h, F_v = ppft2(image, norm=norm)
#     return torch.stack([F_h, F_v], dim=-3)

# def ppft_freq_shift(F: torch.Tensor, drop_ratio: float = 0.4) -> torch.Tensor:
#     """
#     对 PPFT 频谱每条线做低频丢弃 + 高频移位 + 补零。

#     参数:
#         F          : (..., 2N+1, N+1) 复数张量（PPFT 输出）
#         N          : 图像边长（偶数）
#         drop_ratio : 零频两侧各丢弃比例，默认 0.2

#     返回:
#         F_out      : (..., 2N+1, N+1) 复数张量，与输入形状相同
#     """
#     N = F.shape[-1]-1
#     half  = N // 2                         # 零频索引
#     drop  = round(drop_ratio * half)          # 两侧各丢弃的点数
#     keep  = half - drop                    # 两侧各保留的高频点数

#     assert keep >= 0, (
#         f"drop_ratio={drop_ratio} 过大：half={half}, drop={drop}, keep={keep}<0"
#     )

#     line_len = N + 1                       # 每行总长
#     assert F.shape[-1] == line_len, f"最后一维期望 {line_len}，实际 {F.shape[-1]}"

#     # ── 从原始行中取出各段 ────────────────────────────────────
#     # 负高频（左侧高频）: 索引 [0 .. keep-1]         共 keep 个
#     # 丢弃区（左）      : 索引 [keep .. half-1]       共 drop 个  → 不用
#     # 零频              : 索引 [half]                 共 1   个
#     # 丢弃区（右）      : 索引 [half+1 .. half+drop]  共 drop 个  → 不用
#     # 正高频（右侧高频）: 索引 [half+drop+1 .. N]     共 keep 个

#     neg_high = F[..., :keep]                      # (..., 2N+1, keep)
#     dc       = F[..., half : half + 1]            # (..., 2N+1, 1)
#     pos_high = F[..., half + drop + 1 : N + 1]   # (..., 2N+1, keep)

#     # 验证取出的正高频长度
#     assert pos_high.shape[-1] == keep, (
#         f"正高频长度期望 {keep}，实际 {pos_high.shape[-1]}"
#     )

#     # ── 构造补零段 ────────────────────────────────────────────
#     pad_shape = F.shape[:-1] + (drop,)
#     zeros = torch.zeros(pad_shape, dtype=F.dtype, device=F.device)

#     # ── 拼接：[0...0 | neg_high | dc | pos_high | 0...0] ─────
#     # 长度验证: drop + keep + 1 + keep + drop = 2*drop + 2*keep + 1
#     #         = 2*drop + 2*(half-drop) + 1 = 2*half + 1 = N + 1  ✓
#     F_out = torch.cat([zeros, neg_high, dc, pos_high, zeros], dim=-1)

#     assert F_out.shape == F.shape, f"输出形状 {F_out.shape} ≠ 输入形状 {F.shape}"
#     return F_out