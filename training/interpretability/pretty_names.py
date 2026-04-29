"""
interpretability/pretty_names.py
================================
Code-name → publication-ready label mapping for SCM nodes, forensic features,
consistency rules, and curated facial attributes.

All returned labels are matplotlib *mathtext* strings (the `$...$` flavour),
so they render with sub/superscripts and Greek letters using matplotlib's
built-in mathtext engine — **no external LaTeX install required**.

Usage:
    from interpretability.pretty_names import pretty, group_of, GROUP_COLORS

    label = pretty('ff_srm_hedge_kurt')      # → r'$\\kappa_{\\mathrm{SRM}}^{\\mathrm{Hedge}}$'
    grp   = group_of('ff_srm_hedge_kurt')    # → 'srm'
    color = GROUP_COLORS[grp]                # → '#9b59b6'

Glossary (printed in figure captions to keep math dense):
    σ²_Lap = Laplacian variance         (regional blur)
    |∇I|   = Sobel gradient magnitude   (boundary sharpness)
    χ²     = chi-squared histogram dist (colour mismatch)
    E_DCT  = DCT high-frequency energy
    PPNC   = pairwise patch noise consistency
    CCNC   = cross-channel noise consistency (Y/Cr/Cb)
    SRM    = steganalysis rich-model residual filter
    κ, μ, σ= kurtosis / mean / std of the residual / noise field
"""

from __future__ import annotations

# ─── Group → colour (for legend swatches and label tinting) ────────────
GROUP_COLORS = {
    'latent':   '#7f7f7f',   # SAE / causal z
    'curated':  '#1f77b4',   # facial attributes
    'rule':     '#2ca02c',   # consistency rules
    'grad':     '#d62728',   # boundary gradients
    'blur':     '#ff7f0e',   # regional blur
    'sym':      '#bcbd22',   # left-right symmetry
    'color':    '#e377c2',   # colour χ²
    'dct':      '#17becf',   # DCT high-freq
    'quality':  '#8c564b',   # quality metrics
    'ppnc':     '#9467bd',   # PPNC
    'ccnc':     '#3a86ff',   # CCNC
    'srm':      '#9b59b6',   # SRM filters
    'noise':    '#e67e22',   # multi-scale noise
    'fft':      '#16a085',   # FFT spectral
    'other':    '#95a5a6',
}

GROUP_LONGNAMES = {
    'latent':   r'Latent (SAE / SCM) $z$',
    'curated':  'Curated facial attribute',
    'rule':     'Consistency rule',
    'grad':     r'Boundary gradient $|\nabla I|$',
    'blur':     r'Laplacian variance $\sigma^2_{\mathrm{Lap}}$',
    'sym':      'Left-right symmetry',
    'color':    r'Colour $\chi^2$',
    'dct':      r'DCT high-freq energy $E_{\mathrm{DCT}}$',
    'quality':  'Quality metric',
    'ppnc':     'PPNC (patch noise)',
    'ccnc':     'CCNC (Y/Cr/Cb noise)',
    'srm':      'SRM residual',
    'noise':    'Multi-scale noise',
    'fft':      'FFT spectral',
}


# ─── Forensic features (83): explicit per-name table ───────────────────
# Built from forensic_features.py; one entry per declared feature.
_FORENSIC = {
    # Boundary gradients (6)
    'ff_grad_skin_bg':       r'$|\nabla I|_{\mathrm{skin\!\to\!bg}}$',
    'ff_grad_eye_skin':      r'$|\nabla I|_{\mathrm{eye\!\to\!skin}}$',
    'ff_grad_lip_skin':      r'$|\nabla I|_{\mathrm{lip\!\to\!skin}}$',
    'ff_grad_nose_skin':     r'$|\nabla I|_{\mathrm{nose\!\to\!skin}}$',
    'ff_grad_brow_skin':     r'$|\nabla I|_{\mathrm{brow\!\to\!skin}}$',
    'ff_grad_mean_boundary': r'$\overline{|\nabla I|}_{\mathrm{bnd}}$',
    # Regional blur (6)
    'ff_blur_skin':              r'$\sigma^2_{\mathrm{Lap}}(\mathrm{skin})$',
    'ff_blur_eye':               r'$\sigma^2_{\mathrm{Lap}}(\mathrm{eye})$',
    'ff_blur_mouth':             r'$\sigma^2_{\mathrm{Lap}}(\mathrm{mouth})$',
    'ff_blur_nose':              r'$\sigma^2_{\mathrm{Lap}}(\mathrm{nose})$',
    'ff_blur_ratio_eye_skin':    r'$\sigma^2_{\mathrm{Lap}}\!:\!\mathrm{eye/skin}$',
    'ff_blur_ratio_mouth_skin':  r'$\sigma^2_{\mathrm{Lap}}\!:\!\mathrm{mouth/skin}$',
    # Symmetry (4)
    'ff_sym_eye':       r'$\mathrm{Sym}_{\mathrm{eye}}$',
    'ff_sym_mouth':     r'$\mathrm{Sym}_{\mathrm{mouth}}$',
    'ff_sym_cheek':     r'$\mathrm{Sym}_{\mathrm{cheek}}$',
    'ff_sym_jawline':   r'$\mathrm{Sym}_{\mathrm{jaw}}$',
    # Colour χ² (4)
    'ff_color_eye_skin':    r'$\chi^2_{\mathrm{eye,skin}}$',
    'ff_color_mouth_skin':  r'$\chi^2_{\mathrm{mouth,skin}}$',
    'ff_color_nose_skin':   r'$\chi^2_{\mathrm{nose,skin}}$',
    'ff_color_lr_cheek':    r'$\chi^2_{\mathrm{cheek\,L,R}}$',
    # DCT high-freq (6)
    'ff_dct_hf_skin':            r'$E_{\mathrm{DCT}}^{\mathrm{skin}}$',
    'ff_dct_hf_eye':             r'$E_{\mathrm{DCT}}^{\mathrm{eye}}$',
    'ff_dct_hf_mouth':           r'$E_{\mathrm{DCT}}^{\mathrm{mouth}}$',
    'ff_dct_hf_nose':            r'$E_{\mathrm{DCT}}^{\mathrm{nose}}$',
    'ff_dct_ratio_eye_skin':     r'$E_{\mathrm{DCT}}\!:\!\mathrm{eye/skin}$',
    'ff_dct_ratio_mouth_skin':   r'$E_{\mathrm{DCT}}\!:\!\mathrm{mouth/skin}$',
    # Quality (4)
    'ff_quality_antispoof':         r'$Q_{\mathrm{antispoof}}$',
    'ff_quality_det_score':         r'$Q_{\mathrm{det}}$',
    'ff_quality_blendshape_sym':    r'$Q_{\mathrm{bs\,sym}}$',
    'ff_quality_landmark_jitter':   r'$Q_{\mathrm{lm\,jit}}$',
    # PPNC (8)
    'ff_ppnc_outer_eye_mean':  r'$\mu_{\mathrm{PPNC}}^{\mathrm{eye\,out}}$',
    'ff_ppnc_outer_eye_std':   r'$\sigma_{\mathrm{PPNC}}^{\mathrm{eye\,out}}$',
    'ff_ppnc_inner_eye_mean':  r'$\mu_{\mathrm{PPNC}}^{\mathrm{eye\,in}}$',
    'ff_ppnc_inner_eye_std':   r'$\sigma_{\mathrm{PPNC}}^{\mathrm{eye\,in}}$',
    'ff_ppnc_cheek_mean':      r'$\mu_{\mathrm{PPNC}}^{\mathrm{cheek}}$',
    'ff_ppnc_cheek_std':       r'$\sigma_{\mathrm{PPNC}}^{\mathrm{cheek}}$',
    'ff_ppnc_jaw_mean':        r'$\mu_{\mathrm{PPNC}}^{\mathrm{jaw}}$',
    'ff_ppnc_jaw_std':         r'$\sigma_{\mathrm{PPNC}}^{\mathrm{jaw}}$',
    # CCNC (8)
    'ff_ccnc_energy_y':    r'$E_{\mathrm{CCNC}}^{Y}$',
    'ff_ccnc_energy_cr':   r'$E_{\mathrm{CCNC}}^{Cr}$',
    'ff_ccnc_energy_cb':   r'$E_{\mathrm{CCNC}}^{Cb}$',
    'ff_ccnc_corr_y_cr':   r'$\rho_{\mathrm{CCNC}}^{Y\!,\,Cr}$',
    'ff_ccnc_corr_y_cb':   r'$\rho_{\mathrm{CCNC}}^{Y\!,\,Cb}$',
    'ff_ccnc_corr_cr_cb':  r'$\rho_{\mathrm{CCNC}}^{Cr\!,\,Cb}$',
    'ff_ccnc_ratio_y_cr':  r'$E_{\mathrm{CCNC}}^{Y/Cr}$',
    'ff_ccnc_ratio_y_cb':  r'$E_{\mathrm{CCNC}}^{Y/Cb}$',
    # SRM (15) — 5 filters × {μ, σ, κ}
    'ff_srm_hedge_mean':   r'$\mu_{\mathrm{SRM}}^{\mathrm{Hedge}}$',
    'ff_srm_hedge_std':    r'$\sigma_{\mathrm{SRM}}^{\mathrm{Hedge}}$',
    'ff_srm_hedge_kurt':   r'$\kappa_{\mathrm{SRM}}^{\mathrm{Hedge}}$',
    'ff_srm_vedge_mean':   r'$\mu_{\mathrm{SRM}}^{\mathrm{Vedge}}$',
    'ff_srm_vedge_std':    r'$\sigma_{\mathrm{SRM}}^{\mathrm{Vedge}}$',
    'ff_srm_vedge_kurt':   r'$\kappa_{\mathrm{SRM}}^{\mathrm{Vedge}}$',
    'ff_srm_hlap_mean':    r'$\mu_{\mathrm{SRM}}^{\mathrm{HLap}}$',
    'ff_srm_hlap_std':     r'$\sigma_{\mathrm{SRM}}^{\mathrm{HLap}}$',
    'ff_srm_hlap_kurt':    r'$\kappa_{\mathrm{SRM}}^{\mathrm{HLap}}$',
    'ff_srm_square_mean':  r'$\mu_{\mathrm{SRM}}^{\mathrm{Sq}}$',
    'ff_srm_square_std':   r'$\sigma_{\mathrm{SRM}}^{\mathrm{Sq}}$',
    'ff_srm_square_kurt':  r'$\kappa_{\mathrm{SRM}}^{\mathrm{Sq}}$',
    'ff_srm_diag_mean':    r'$\mu_{\mathrm{SRM}}^{\mathrm{Diag}}$',
    'ff_srm_diag_std':     r'$\sigma_{\mathrm{SRM}}^{\mathrm{Diag}}$',
    'ff_srm_diag_kurt':    r'$\kappa_{\mathrm{SRM}}^{\mathrm{Diag}}$',
    # Multi-scale noise (12)
    'ff_noise_s1_mean':         r'$\mu_{\mathrm{noise}}^{(1)}$',
    'ff_noise_s1_std':          r'$\sigma_{\mathrm{noise}}^{(1)}$',
    'ff_noise_s1_kurt':         r'$\kappa_{\mathrm{noise}}^{(1)}$',
    'ff_noise_s2_mean':         r'$\mu_{\mathrm{noise}}^{(2)}$',
    'ff_noise_s2_std':          r'$\sigma_{\mathrm{noise}}^{(2)}$',
    'ff_noise_s2_kurt':         r'$\kappa_{\mathrm{noise}}^{(2)}$',
    'ff_noise_s4_mean':         r'$\mu_{\mathrm{noise}}^{(4)}$',
    'ff_noise_s4_std':          r'$\sigma_{\mathrm{noise}}^{(4)}$',
    'ff_noise_s4_kurt':         r'$\kappa_{\mathrm{noise}}^{(4)}$',
    'ff_noise_xcorr_s1s2':      r'$\rho_{\mathrm{noise}}^{1,2}$',
    'ff_noise_xcorr_s2s4':      r'$\rho_{\mathrm{noise}}^{2,4}$',
    'ff_noise_fine_coarse_ratio': r'$E_{\mathrm{noise}}^{\mathrm{fine/coarse}}$',
    # FFT (10)
    'ff_fft_bin0':       r'$\mathrm{FFT}_{0}$',
    'ff_fft_bin1':       r'$\mathrm{FFT}_{1}$',
    'ff_fft_bin2':       r'$\mathrm{FFT}_{2}$',
    'ff_fft_bin3':       r'$\mathrm{FFT}_{3}$',
    'ff_fft_bin4':       r'$\mathrm{FFT}_{4}$',
    'ff_fft_bin5':       r'$\mathrm{FFT}_{5}$',
    'ff_fft_bin6':       r'$\mathrm{FFT}_{6}$',
    'ff_fft_bin7':       r'$\mathrm{FFT}_{7}$',
    'ff_fft_slope':      r'$\beta_{\mathrm{FFT}}$',
    'ff_fft_hf_ratio':   r'$r_{\mathrm{FFT}}^{\mathrm{HF}}$',
}


# ─── Curated facial attributes (fs_*) — generic prettifier ─────────────
_CURATED_OVERRIDES = {
    'fs_gender_score':              r'$\mathrm{gender}$',
    'fs_age_score':                 r'$\mathrm{age}$',
    'fs_jaw_width_ratio':           r'$\mathrm{jaw\,w/h}$',
    'fs_nose_width_ratio':          r'$\mathrm{nose\,w/h}$',
    'fs_face_width_height_ratio':   r'$\mathrm{face\,w/h}$',
    'fs_interpupillary_ratio':      r'$\mathrm{IPD/face_w}$',
    'fs_chin_angle':                r'$\theta_{\mathrm{chin}}$',
    'fs_brow_height_ratio':         r'$\mathrm{brow\,h}$',
    'fs_eye_lr_symmetry':           r'$\mathrm{Sym}_{\mathrm{eye\,LR}}$',
    'fs_jaw_symmetry':              r'$\mathrm{Sym}_{\mathrm{jaw}}$',
    'fs_det_confidence':            r'$Q_{\mathrm{det}}$',
    'fs_pose_yaw':                  r'$\theta_{\mathrm{yaw}}$',
    'fs_pose_pitch':                r'$\theta_{\mathrm{pitch}}$',
    'fs_gaze_left_x':               r'$\mathrm{gaze}_{L,x}$',
    'fs_gaze_right_x':              r'$\mathrm{gaze}_{R,x}$',
    'fs_gaze_left_y':               r'$\mathrm{gaze}_{L,y}$',
    'fs_expr_happy':                r'$\mathrm{expr.\,happy}$',
    'fs_expr_sad':                  r'$\mathrm{expr.\,sad}$',
    'fs_expr_angry':                r'$\mathrm{expr.\,angry}$',
    'fs_expr_surprise':             r'$\mathrm{expr.\,surprise}$',
    'fs_AU1':                       r'$\mathrm{AU1}$',
    'fs_AU4':                       r'$\mathrm{AU4}$',
    'fs_AU6':                       r'$\mathrm{AU6}$',
    'fs_AU9':                       r'$\mathrm{AU9}$',
    'fs_AU12':                      r'$\mathrm{AU12}$',
    'fs_AU15':                      r'$\mathrm{AU15}$',
}


# ─── Consistency rules — keep cr_ prefix as "Rule:" tag ────────────────
_RULE_OVERRIDES = {
    'cr_mutual_mouth':             r'$\mathrm{R}_{\mathrm{mouth\,open\!\times\!closed}}$',
    'cr_mutual_gender':            r'$\mathrm{R}_{\mathrm{male\!\times\!female}}$',
    'cr_mutual_smile_frown':       r'$\mathrm{R}_{\mathrm{smile\!\times\!frown}}$',
    'cr_lighting_conflict':        r'$\mathrm{R}_{\mathrm{light\!\times\!dim}}$',
    'cr_happy_au6':                r'$\mathrm{R}_{\mathrm{happy\!-\!AU6}}$',
    'cr_happy_au12':               r'$\mathrm{R}_{\mathrm{happy\!-\!AU12}}$',
    'cr_surprise_au1au2':          r'$\mathrm{R}_{\mathrm{surp.\!-\!AU1,2}}$',
    'cr_sad_au15':                 r'$\mathrm{R}_{\mathrm{sad\!-\!AU15}}$',
    'cr_angry_au4':                r'$\mathrm{R}_{\mathrm{angry\!-\!AU4}}$',
    'cr_fear_au1au5':              r'$\mathrm{R}_{\mathrm{fear\!-\!AU1,5}}$',
    'cr_disgust_au9':              r'$\mathrm{R}_{\mathrm{disg.\!-\!AU9}}$',
    'cr_contempt_au14':            r'$\mathrm{R}_{\mathrm{cont.\!-\!AU14}}$',
    'cr_neutral_any_au':           r'$\mathrm{R}_{\mathrm{neutral\!\times\!AU}}$',
    'cr_double_chin_narrow_jaw':   r'$\mathrm{R}_{\mathrm{dbl.chin\!\times\!narrow}}$',
    'cr_square_face_narrow_jaw':   r'$\mathrm{R}_{\mathrm{square\!\times\!narrow}}$',
    'cr_round_face_pointed_chin':  r'$\mathrm{R}_{\mathrm{round\!\times\!pointed}}$',
    'cr_skin_tone_conflict':       r'$\mathrm{R}_{\mathrm{fair\!\times\!dark}}$',
    'cr_skin_age_acne':            r'$\mathrm{R}_{\mathrm{acne\!\times\!elderly}}$',
    'cr_symmetry_conflict':        r'$\mathrm{R}_{\mathrm{sym\!\times\!asym}}$',
    'cr_image_quality':            r'$\mathrm{R}_{\mathrm{blur\!\times\!sharp}}$',
    'cr_eye_asymmetry':            r'$\mathrm{R}_{\mathrm{eye\,asym}}$',
    'cr_jaw_asymmetry':            r'$\mathrm{R}_{\mathrm{jaw\,asym}}$',
    'cr_gaze_lr_divergence':       r'$\mathrm{R}_{\mathrm{gaze\,LR}}$',
}


# ─── Public API ────────────────────────────────────────────────────────

def group_of(name: str) -> str:
    """Return the semantic group (used for colouring / legend)."""
    if not isinstance(name, str):
        return 'other'
    if name.startswith('z_') or name.startswith('z'):
        if name[1:].lstrip('_').isdigit():
            return 'latent'
    if name.startswith('cr_'):
        return 'rule'
    if name.startswith('fs_'):
        return 'curated'
    if name.startswith('ff_grad'):    return 'grad'
    if name.startswith('ff_blur'):    return 'blur'
    if name.startswith('ff_sym'):     return 'sym'
    if name.startswith('ff_color'):   return 'color'
    if name.startswith('ff_dct'):     return 'dct'
    if name.startswith('ff_quality'): return 'quality'
    if name.startswith('ff_ppnc'):    return 'ppnc'
    if name.startswith('ff_ccnc'):    return 'ccnc'
    if name.startswith('ff_srm'):     return 'srm'
    if name.startswith('ff_noise'):   return 'noise'
    if name.startswith('ff_fft'):     return 'fft'
    return 'other'


def pretty(name: str) -> str:
    """
    Translate a code-style node name into a publication label.

    Returns a matplotlib mathtext string (`$...$`). For unknown names
    the original is wrapped in `\\mathrm{}` so it still renders cleanly.
    """
    if not isinstance(name, str):
        return str(name)

    # SAE / SCM latent dims
    if name.startswith('z_'):
        idx = name[2:]
        if idx.isdigit():
            return rf'$z_{{{idx}}}$'
    if name in _FORENSIC:
        return _FORENSIC[name]
    if name in _CURATED_OVERRIDES:
        return _CURATED_OVERRIDES[name]
    if name in _RULE_OVERRIDES:
        return _RULE_OVERRIDES[name]

    # Generic fallbacks for unknown but prefixed names
    if name.startswith('fs_'):
        return r'$\mathrm{' + name[3:].replace('_', r'\,') + r'}$'
    if name.startswith('cr_'):
        return r'$\mathrm{R}_{\mathrm{' + name[3:].replace('_', r'\,') + r'}}$'
    if name.startswith('ff_'):
        return r'$\mathrm{' + name[3:].replace('_', r'\,') + r'}$'
    return r'$\mathrm{' + name.replace('_', r'\,') + r'}$'


def pretty_many(names):
    """Vectorised `pretty` for a list-like of names."""
    return [pretty(n) for n in names]
