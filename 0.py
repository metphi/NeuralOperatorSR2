"""SRMetric 类的验证脚本。

测试覆盖：
1. 基础调用：numpy / tensor 都能正常跑通
2. 一致性：numpy 版和 pt 版在相同输入下结果是否一致
3. 边界情况：相同图（PSNR=inf, SSIM=1）、完全不同的图
4. 输入形式：3D tensor / 4D tensor 都能用、batch 维正确
5. 配置切换：Y 通道 vs RGB、PSNR vs SSIM
6. 错误捕获：类型不匹配、形状不对、非法参数
7. device 自适应：CPU tensor 结果保持在 CPU（GPU 测试跳过，环境无 GPU）
"""

import numpy as np
import torch

from utils.merics_utlis import SRMetric


# ============================================================
# 辅助工具
# ============================================================

def make_numpy_pair(h=128, w=128, noise_std=10.0, seed=0):
    """生成一对 numpy 图：一张"GT"，一张加了噪声的"SR"。
    
    返回：BGR, uint8, HWC, [0, 255] —— 符合 cv2.imread 的输出规范。
    """
    rng = np.random.RandomState(seed)
    gt = rng.randint(0, 256, size=(h, w, 3), dtype=np.uint8)
    noise = rng.randn(h, w, 3) * noise_std
    sr = np.clip(gt.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return sr, gt


def numpy_bgr_to_tensor_rgb(img_bgr_uint8):
    """把 cv2 风格的 numpy 图 (HWC, BGR, uint8) 转成 pt 版期望的
    tensor (1, 3, H, W), RGB, float, [0, 1]。
    """
    img_rgb = img_bgr_uint8[:, :, ::-1].copy()       # BGR -> RGB
    img_float = img_rgb.astype(np.float32) / 255.0   # [0, 255] -> [0, 1]
    tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0)  # HWC -> NCHW
    return tensor


def banner(title):
    print('\n' + '=' * 70)
    print(f'  {title}')
    print('=' * 70)


def check(cond, msg):
    status = '✅ PASS' if cond else '❌ FAIL'
    print(f'  {status}: {msg}')
    if not cond:
        raise AssertionError(msg)


# ============================================================
# 测试 1：基础调用
# ============================================================

def test_basic_numpy():
    banner('测试 1：基础 numpy 调用')
    sr, gt = make_numpy_pair(seed=42)
    print(f'  输入形状: sr={sr.shape} ({sr.dtype}), gt={gt.shape} ({gt.dtype})')

    psnr_y = SRMetric('psnr', test_y_channel=True, crop_border=4)
    ssim_y = SRMetric('ssim', test_y_channel=True, crop_border=4)
    psnr_rgb = SRMetric('psnr', test_y_channel=False, crop_border=4)
    ssim_rgb = SRMetric('ssim', test_y_channel=False, crop_border=4)

    p_y = psnr_y(sr, gt)
    s_y = ssim_y(sr, gt)
    p_rgb = psnr_rgb(sr, gt)
    s_rgb = ssim_rgb(sr, gt)

    print(f'  {psnr_y!r} → {p_y:.4f}')
    print(f'  {ssim_y!r} → {s_y:.4f}')
    print(f'  {psnr_rgb!r} → {p_rgb:.4f}')
    print(f'  {ssim_rgb!r} → {s_rgb:.4f}')

    check(isinstance(p_y, float), 'numpy PSNR 返回 Python float')
    check(isinstance(s_y, (float, np.floating)), 'numpy SSIM 返回 float/np.floating')
    check(20 < p_y < 50, f'PSNR 值在合理范围 (20-50 dB)，实际 {p_y:.2f}')
    check(0 < s_y < 1, f'SSIM 值在 (0, 1)，实际 {s_y:.4f}')


def test_basic_tensor():
    banner('测试 2：基础 tensor 调用（4D）')
    sr_np, gt_np = make_numpy_pair(seed=42)
    sr_t = numpy_bgr_to_tensor_rgb(sr_np)
    gt_t = numpy_bgr_to_tensor_rgb(gt_np)
    print(f'  输入形状: sr={tuple(sr_t.shape)} ({sr_t.dtype}), '
          f'gt={tuple(gt_t.shape)} ({gt_t.dtype})')

    psnr_y = SRMetric('psnr', test_y_channel=True, crop_border=4)
    ssim_y = SRMetric('ssim', test_y_channel=True, crop_border=4)

    p = psnr_y(sr_t, gt_t)
    s = ssim_y(sr_t, gt_t)

    print(f'  PSNR → {p}')
    print(f'  SSIM → {s}')

    check(isinstance(p, torch.Tensor), 'tensor PSNR 返回 torch.Tensor')
    check(isinstance(s, torch.Tensor), 'tensor SSIM 返回 torch.Tensor')
    check(p.shape == (1,), f'单张图 batch 的返回 shape 是 (1,)，实际 {tuple(p.shape)}')
    check(s.shape == (1,), f'单张图 batch 的返回 shape 是 (1,)，实际 {tuple(s.shape)}')


# ============================================================
# 测试 3：numpy 和 pt 版本数值一致性（关键）
# ============================================================

def test_numpy_pt_consistency():
    banner('测试 3：numpy 版 vs pt 版 数值一致性')
    sr_np, gt_np = make_numpy_pair(seed=123)
    # 关键：numpy 输入是 BGR，tensor 输入是 RGB
    # 两者指向"同一张图"，所以 Y 通道上算出来的 PSNR/SSIM 应该一致
    sr_t = numpy_bgr_to_tensor_rgb(sr_np)
    gt_t = numpy_bgr_to_tensor_rgb(gt_np)

    for metric_name in ['psnr', 'ssim']:
        for y_channel in [True, False]:
            m = SRMetric(metric_name, test_y_channel=y_channel, crop_border=4)
            v_np = m(sr_np, gt_np)
            v_pt = m(sr_t, gt_t).item()

            diff = abs(v_np - v_pt)
            tag = 'Y' if y_channel else 'RGB'
            print(f'  [{metric_name.upper():4s} | {tag:3s}] '
                  f'numpy={v_np:.6f}, pt={v_pt:.6f}, diff={diff:.2e}')

            # 两种实现（cv2.filter2D vs F.conv2d、uint8 量化 vs float 直算）
            # 天然有一点点差异，1e-3 是经验上合理的容忍度
            tol = 1e-3 if metric_name == 'psnr' else 1e-3
            check(diff < tol, f'{metric_name.upper()} ({tag}) 差异 < {tol}')


# ============================================================
# 测试 4：边界情况
# ============================================================

def test_identical_images():
    banner('测试 4：完全相同的图（PSNR=inf, SSIM=1）')
    sr, gt = make_numpy_pair(seed=7)
    
    psnr_m = SRMetric('psnr', test_y_channel=True, crop_border=4)
    ssim_m = SRMetric('ssim', test_y_channel=True, crop_border=4)

    # numpy: 相同的图
    p_np = psnr_m(gt, gt)
    s_np = ssim_m(gt, gt)
    print(f'  numpy: PSNR={p_np}, SSIM={s_np:.6f}')
    check(p_np == float('inf'), 'numpy 相同图 PSNR 应为 inf')
    check(abs(s_np - 1.0) < 1e-6, f'numpy 相同图 SSIM 应为 1.0，实际 {s_np}')

    # tensor: 相同的图
    gt_t = numpy_bgr_to_tensor_rgb(gt)
    p_pt = psnr_m(gt_t, gt_t).item()
    s_pt = ssim_m(gt_t, gt_t).item()
    print(f'  tensor: PSNR={p_pt:.2f}, SSIM={s_pt:.6f}')
    # pt 版里有 +1e-8 防止除零，所以相同图 PSNR 不是 inf 而是很大的有限值
    check(p_pt > 70, f'tensor 相同图 PSNR 应为很大的有限值 (>70 dB)，实际 {p_pt:.2f}')
    check(abs(s_pt - 1.0) < 1e-6, f'tensor 相同图 SSIM 应为 1.0，实际 {s_pt}')


def test_very_different_images():
    banner('测试 5：差异很大的图（低 PSNR, 低 SSIM）')
    rng = np.random.RandomState(999)
    # 两张完全独立的随机噪声图
    img_a = rng.randint(0, 256, size=(128, 128, 3), dtype=np.uint8)
    img_b = rng.randint(0, 256, size=(128, 128, 3), dtype=np.uint8)

    psnr_m = SRMetric('psnr', test_y_channel=True, crop_border=0)
    ssim_m = SRMetric('ssim', test_y_channel=True, crop_border=0)

    p = psnr_m(img_a, img_b)
    s = ssim_m(img_a, img_b)
    print(f'  独立随机噪声图: PSNR={p:.2f}, SSIM={s:.4f}')
    check(p < 15, f'独立噪声的 PSNR 应很低 (<15)，实际 {p:.2f}')
    check(s < 0.1, f'独立噪声的 SSIM 应接近 0，实际 {s:.4f}')


# ============================================================
# 测试 6：batch 处理
# ============================================================

def test_batch_tensor():
    banner('测试 6：batch tensor（N>1）')
    # 构造一个 batch=4 的测试：每张图噪声强度不同，PSNR 应当递减
    batch_sr, batch_gt = [], []
    for i, noise_std in enumerate([2.0, 5.0, 10.0, 20.0]):
        sr_np, gt_np = make_numpy_pair(noise_std=noise_std, seed=i)
        batch_sr.append(numpy_bgr_to_tensor_rgb(sr_np))
        batch_gt.append(numpy_bgr_to_tensor_rgb(gt_np))

    sr_batch = torch.cat(batch_sr, dim=0)  # (4, 3, H, W)
    gt_batch = torch.cat(batch_gt, dim=0)
    print(f'  batch 形状: {tuple(sr_batch.shape)}')

    psnr_m = SRMetric('psnr', test_y_channel=True, crop_border=4)
    scores = psnr_m(sr_batch, gt_batch)
    print(f'  每张图的 PSNR: {[f"{x:.2f}" for x in scores.tolist()]}')

    check(scores.shape == (4,), f'batch=4 应返回 shape (4,)，实际 {tuple(scores.shape)}')
    # 噪声递增 → PSNR 递减
    diffs = scores[:-1] - scores[1:]  # 每一项应 > 0
    check((diffs > 0).all().item(),
          f'噪声递增时 PSNR 应单调递减，实际差值 {diffs.tolist()}')


def test_3d_tensor():
    banner('测试 7：3D tensor（C,H,W），自动补 batch 维')
    sr_np, gt_np = make_numpy_pair(seed=11)
    sr_t_4d = numpy_bgr_to_tensor_rgb(sr_np)
    gt_t_4d = numpy_bgr_to_tensor_rgb(gt_np)
    sr_t_3d = sr_t_4d.squeeze(0)
    gt_t_3d = gt_t_4d.squeeze(0)
    print(f'  3D 输入形状: {tuple(sr_t_3d.shape)}')

    psnr_m = SRMetric('psnr', test_y_channel=True, crop_border=4)
    v_3d = psnr_m(sr_t_3d, gt_t_3d)
    v_4d = psnr_m(sr_t_4d, gt_t_4d)
    print(f'  3D 输入 → {v_3d}')
    print(f'  4D 输入 → {v_4d}')
    check(torch.allclose(v_3d, v_4d), '3D 和 4D 输入结果应一致')


# ============================================================
# 测试 8：device 自适应
# ============================================================

def test_device_awareness():
    banner('测试 8：device 自适应')
    sr_np, gt_np = make_numpy_pair(seed=55)
    sr_t = numpy_bgr_to_tensor_rgb(sr_np)
    gt_t = numpy_bgr_to_tensor_rgb(gt_np)

    psnr_m = SRMetric('psnr', test_y_channel=True, crop_border=4)

    # CPU 测试
    v_cpu = psnr_m(sr_t, gt_t)
    print(f'  CPU 输入 → 返回 device={v_cpu.device}, value={v_cpu.item():.4f}')
    check(v_cpu.device.type == 'cpu', 'CPU tensor 输入应返回 CPU tensor')

    # GPU 测试（如可用）
    if torch.cuda.is_available():
        sr_gpu = sr_t.cuda()
        gt_gpu = gt_t.cuda()
        v_gpu = psnr_m(sr_gpu, gt_gpu)
        print(f'  GPU 输入 → 返回 device={v_gpu.device}, value={v_gpu.item():.4f}')
        check(v_gpu.device.type == 'cuda', 'GPU tensor 输入应返回 GPU tensor')
        check(abs(v_cpu.item() - v_gpu.item()) < 1e-4, 'CPU / GPU 结果应一致')
    else:
        print('  (无 GPU，跳过 CUDA 测试)')


# ============================================================
# 测试 9：错误处理
# ============================================================

def test_error_handling():
    banner('测试 9：错误处理')

    # 9.1 非法 metric
    try:
        SRMetric('mse')
        check(False, '非法 metric 应抛 ValueError')
    except ValueError as e:
        print(f'  ✅ 非法 metric 抛 ValueError: {e}')

    # 9.2 非法 input_order
    try:
        SRMetric('psnr', input_order='WHC')
        check(False, '非法 input_order 应抛 ValueError')
    except ValueError as e:
        print(f'  ✅ 非法 input_order 抛 ValueError: {e}')

    # 9.3 类型不一致
    sr_np, gt_np = make_numpy_pair(seed=1)
    sr_t = numpy_bgr_to_tensor_rgb(sr_np)
    psnr_m = SRMetric('psnr')
    try:
        psnr_m(sr_np, sr_t)
        check(False, '类型不一致应抛 TypeError')
    except TypeError as e:
        print(f'  ✅ 类型不一致抛 TypeError: {e}')

    # 9.4 不支持的类型（list）
    try:
        psnr_m([1, 2, 3], [4, 5, 6])
        check(False, 'list 输入应抛 TypeError')
    except TypeError as e:
        print(f'  ✅ list 输入抛 TypeError: {e}')

    # 9.5 tensor 维度错误（2D）
    try:
        bad = torch.randn(128, 128)
        psnr_m(bad, bad)
        check(False, '2D tensor 应抛 ValueError')
    except ValueError as e:
        print(f'  ✅ 2D tensor 抛 ValueError: {e}')


# ============================================================
# 测试 10：__repr__
# ============================================================

def test_repr():
    banner('测试 10：__repr__')
    m1 = SRMetric('psnr', test_y_channel=True, crop_border=4)
    m2 = SRMetric('ssim', test_y_channel=False, crop_border=0)
    print(f'  {m1!r}')
    print(f'  {m2!r}')
    check("metric='psnr'" in repr(m1), 'repr 含 metric 字段')
    check("test_y_channel=False" in repr(m2), 'repr 含 test_y_channel 字段')


# ============================================================
# 主入口
# ============================================================

if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)

    tests = [
        test_basic_numpy,
        test_basic_tensor,
        test_numpy_pt_consistency,
        test_identical_images,
        test_very_different_images,
        test_batch_tensor,
        test_3d_tensor,
        test_device_awareness,
        test_error_handling,
        test_repr,
    ]

    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f'\n❌ 测试 {t.__name__} 失败: {type(e).__name__}: {e}')

    print('\n' + '=' * 70)
    print(f'  总计：{passed}/{len(tests)} 测试通过')
    print('=' * 70)