"""
networks/nesy_defake/forensic_gpu.py
====================================
Differentiable (torch) re-implementation of the pixel-level forensic features
(Phase-3 Task 4). SPEC + TEST ONLY — this is NOT wired into any training path.
Integration is a later decision gated on the T3 shift report and the f2 vs
f2_noaug result.

Operates on `(B, 3, H, W)` images in `[0, 1]`, RGB. Reproduces the LANDMARK-FREE
forensic groups of preprocessing/forensic_helpers.py:

  patch_noise  (30-45): PPNC (fixed symmetric patches) + CCNC (YCrCb cross-channel)
  srm_noise    (46-72): 5 SRM steganalysis kernels + 3-scale LoG noise residuals
  fft_spectral (73-82): radial FFT magnitude spectrum

PARITY-IMPOSSIBLE groups (return NaN — documented, not a bug):
  boundary_texture (0-11), symmetry_color (12-29)
    → every feature is defined over SegFormer face-parsing regions (skin / eye /
      mouth / nose / brow masks). Without the parsing map these cannot be
      reproduced from pixels. Features 26-29 (quality_*) additionally require the
      anti-spoof / face-detector model scores (they are 0 in the cache unless
      those models ran).

Numerical-parity notes (why cosine, not bit-exact):
  * gray/YCrCb use the cv2 coefficients but skip cv2's uint8 rounding;
  * Gaussian blur uses cv2's ksize rule `round(6σ+1)|1` with reflect-101 pad;
  * std is population (ddof=0) to match numpy; kurtosis is excess kurtosis;
  * features depending on resolution (spectral bins, SRM stats) match only when
    run at the SAME size the cache used (FF++ frames are 256×256, NOT 224).
"""
import math

import torch
import torch.nn.functional as F

N_FORENSIC = 83

# Group name → (start, end) — mirrors ccv_branch.FORENSIC_GROUPS.
GROUPS = {
    'boundary_texture': (0, 12),
    'symmetry_color':   (12, 30),
    'patch_noise':      (30, 46),
    'srm_noise':        (46, 73),
    'fft_spectral':     (73, 83),
}
REPRODUCIBLE = ('patch_noise', 'srm_noise', 'fft_spectral')

_SRM_KERNELS = torch.tensor([
    [[0, 0, 0], [0, -1, 1], [0, 0, 0]],
    [[0, 0, 0], [0, -1, 0], [0, 1, 0]],
    [[0, 0, 0], [1, -2, 1], [0, 0, 0]],
    [[-1, 2, -1], [2, -4, 2], [-1, 2, -1]],
    [[0, 0, -1], [0, 2, 0], [-1, 0, 0]],
], dtype=torch.float32)

_PPNC_PAIRS = [(0.20, 0.35, 0.80, 0.35), (0.35, 0.35, 0.65, 0.35),
               (0.20, 0.55, 0.80, 0.55), (0.25, 0.80, 0.75, 0.80)]


# ── primitives ──────────────────────────────────────────────────────────────

def rgb_to_gray255(x):
    """(B,3,H,W) in [0,1] RGB → (B,H,W) gray in [0,255] (cv2 coefficients)."""
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    return (0.299 * r + 0.587 * g + 0.114 * b) * 255.0


def rgb_to_ycrcb255(x):
    """(B,3,H,W) [0,1] RGB → Y,Cr,Cb each (B,H,W) in [0,255] (cv2 formula)."""
    r, g, b = x[:, 0] * 255.0, x[:, 1] * 255.0, x[:, 2] * 255.0
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cr = (r - y) * 0.713 + 128.0
    cb = (b - y) * 0.564 + 128.0
    return y, cr, cb


def _gauss_kernel(sigma, device, dtype):
    ksize = int(round(sigma * 6 + 1)) | 1        # cv2 ksize=0 rule for 8U
    half = ksize // 2
    xs = torch.arange(-half, half + 1, device=device, dtype=dtype)
    k = torch.exp(-(xs ** 2) / (2 * sigma ** 2))
    return (k / k.sum())


def gaussian_blur(img, sigma):
    """cv2.GaussianBlur(σ) equivalent — separable, reflect-101 pad. img (B,H,W)."""
    k = _gauss_kernel(sigma, img.device, img.dtype)
    ksize = k.numel(); half = ksize // 2
    x = img.unsqueeze(1)                          # (B,1,H,W)
    x = F.pad(x, (half, half, half, half), mode='reflect')
    kh = k.view(1, 1, 1, ksize)
    kv = k.view(1, 1, ksize, 1)
    x = F.conv2d(x, kh)
    x = F.conv2d(x, kv)
    return x.squeeze(1)


def _filter2d_reflect(img, kernel):
    """cv2.filter2D (correlation, reflect-101 pad). img (B,H,W), kernel (kh,kw)."""
    kh, kw = kernel.shape
    x = img.unsqueeze(1)
    x = F.pad(x, (kw // 2, kw // 2, kh // 2, kh // 2), mode='reflect')
    return F.conv2d(x, kernel.view(1, 1, kh, kw)).squeeze(1)


def _pop_std(x, dim):
    return x.var(dim=dim, unbiased=False).clamp_min(0).sqrt()


def _excess_kurtosis(x, dim):
    m = x.mean(dim=dim, keepdim=True)
    c = x - m
    s2 = (c ** 2).mean(dim=dim)
    m4 = (c ** 4).mean(dim=dim)
    return torch.where(s2 < 1e-16, torch.zeros_like(s2), m4 / (s2 ** 2) - 3.0)


def _flat_corr(a, b):
    """Pearson corr over the last flattened dims, per batch. a,b (B,H,W)."""
    a = a.flatten(1); b = b.flatten(1)
    a = a - a.mean(1, keepdim=True); b = b - b.mean(1, keepdim=True)
    num = (a * b).sum(1)
    den = torch.sqrt((a * a).sum(1) * (b * b).sum(1))
    return torch.where(den > 1e-8, num / den, torch.zeros_like(num))


# ── groups ──────────────────────────────────────────────────────────────────

def group_ppnc(gray):
    """PPNC (8-d): symmetric-patch noise mean/std differences."""
    B, H, W = gray.shape
    patch_r = max(int(min(H, W) * 0.08), 4)
    noise = gray - gaussian_blur(gray, 2.0)
    feats = []
    for lcx, lcy, rcx, rcy in _PPNC_PAIRS:
        lx, ly = int(lcx * W), int(lcy * H)
        rx, ry = int(rcx * W), int(rcy * H)

        def patch(cx, cy):
            y0, y1 = max(0, cy - patch_r), min(H, cy + patch_r)
            x0, x1 = max(0, cx - patch_r), min(W, cx + patch_r)
            return noise[:, y0:y1, x0:x1].flatten(1)

        lp, rp = patch(lx, ly), patch(rx, ry)
        feats.append((lp.mean(1) - rp.mean(1)).abs())
        feats.append((_pop_std(lp, 1) - _pop_std(rp, 1)).abs())
    return torch.stack(feats, dim=1)              # (B,8)


def group_ccnc(x):
    """CCNC (8-d): YCrCb noise energies, cross-channel corr, luma/chroma ratios."""
    chans = rgb_to_ycrcb255(x)
    noises = [ch - gaussian_blur(ch, 2.0) for ch in chans]
    energies = [(n ** 2).mean(dim=(1, 2)) for n in noises]
    feats = list(energies)
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        feats.append(_flat_corr(noises[i], noises[j]))
    feats.append(energies[0] / energies[1].clamp_min(1e-8))
    feats.append(energies[0] / energies[2].clamp_min(1e-8))
    return torch.stack(feats, dim=1)              # (B,8)


def group_srm(gray):
    """SRM (15-d): 5 kernels × (mean|·|, std, excess-kurtosis)."""
    feats = []
    for k in range(_SRM_KERNELS.shape[0]):
        res = _filter2d_reflect(gray, _SRM_KERNELS[k].to(gray))
        r = res.flatten(1)
        feats.append(r.abs().mean(1))
        feats.append(_pop_std(r, 1))
        feats.append(_excess_kurtosis(r, 1))
    return torch.stack(feats, dim=1)              # (B,15)


def group_multiscale(gray):
    """Multi-scale LoG noise (12-d): 3 scales × stats + xcorr + fine/coarse."""
    feats, res = [], []
    for sigma in (1.0, 2.0, 4.0):
        n = gray - gaussian_blur(gray, sigma)
        res.append(n)
        r = n.flatten(1)
        feats.append(r.abs().mean(1))
        feats.append(_pop_std(r, 1))
        feats.append(_excess_kurtosis(r, 1))
    feats.append(_flat_corr(res[0], res[1]))
    feats.append(_flat_corr(res[1], res[2]))
    e_fine = (res[0] ** 2).mean(dim=(1, 2))
    e_coarse = (res[2] ** 2).mean(dim=(1, 2))
    feats.append(e_fine / e_coarse.clamp_min(1e-8))
    return torch.stack(feats, dim=1)              # (B,12)


def group_spectral(gray, n_bins=8):
    """Radial FFT magnitude spectrum (10-d): 8 radial bins + slope + hf ratio."""
    B, H, W = gray.shape
    img_f = gray / 255.0
    mag = torch.log1p(torch.fft.fftshift(
        torch.fft.fft2(img_f), dim=(-2, -1)).abs())
    cy, cx = H // 2, W // 2
    max_r = min(cy, cx)
    yy, xx = torch.meshgrid(
        torch.arange(H, device=gray.device, dtype=gray.dtype),
        torch.arange(W, device=gray.device, dtype=gray.dtype), indexing='ij')
    r = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    edges = torch.linspace(0, max_r, n_bins + 1, device=gray.device, dtype=gray.dtype)
    idx = torch.bucketize(r.flatten(), edges[1:], right=False).clamp(0, n_bins - 1)
    profile = []
    magf = mag.flatten(1)
    for b in range(n_bins):
        m = (idx == b)
        cnt = m.sum().clamp_min(1)
        profile.append((magf[:, m].sum(1) / cnt))
    profile = torch.stack(profile, dim=1)         # (B, n_bins)

    # spectral slope: least-squares slope of profile vs log(freqs)
    freqs = torch.arange(1, n_bins + 1, device=gray.device, dtype=gray.dtype)
    lf = torch.log(freqs)
    lf_c = lf - lf.mean()
    prof_c = profile - profile.mean(1, keepdim=True)
    slope = (prof_c * lf_c).sum(1) / (lf_c * lf_c).sum().clamp_min(1e-12)
    std_ok = _pop_std(profile, 1) > 1e-8
    slope = torch.where(std_ok, slope, torch.zeros_like(slope))

    total_e = profile.sum(1) + 1e-8
    hf_e = profile[:, n_bins * 3 // 4:].sum(1)
    hf_ratio = hf_e / total_e
    return torch.cat([profile, slope.unsqueeze(1), hf_ratio.unsqueeze(1)], dim=1)


# ── module ──────────────────────────────────────────────────────────────────

class ForensicBankGPU(torch.nn.Module):
    """Differentiable forensic bank. forward(x)->(B,83); region groups are NaN."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gray = rgb_to_gray255(x)
        out = torch.full((x.shape[0], N_FORENSIC), float('nan'),
                         device=x.device, dtype=x.dtype)
        out[:, 30:38] = group_ppnc(gray)
        out[:, 38:46] = group_ccnc(x)
        out[:, 46:61] = group_srm(gray)
        out[:, 61:73] = group_multiscale(gray)
        out[:, 73:83] = group_spectral(gray)
        return out
