"""
Occlusion-aware UWB noise generation script.
Transfers CustomDataset's real UWB noise characteristics to OTB100/UAV123.

Four occlusion zones with independent noise models:
  - Baseline (visible, far from occlusion)
  - Pre-occlusion (visible, within 30 frames before occlusion)
  - Occlusion (occ=1)
  - Post-occlusion (visible, within 30 frames after occlusion)
"""
import numpy as np
import pandas as pd
import os
import argparse
import json
from typing import Optional, Tuple


# ============================================================
# Noise Parameters (estimated from CustomDataset)
# ============================================================

ZONE_PARAMS = {
    'baseline': {
        'phi_u': 0.966, 'sigma_eps_u': 5.0,
        'phi_v': 0.980, 'sigma_eps_v': 3.5,
        'lognorm_mu': 3.0, 'lognorm_sigma': 0.6,
    },
    'pre_occlusion': {
        'phi_u': 0.89, 'sigma_eps_u': 10.0,
        'phi_v': 0.92, 'sigma_eps_v': 9.0,
        'mix_prob': 0.15,
        'extreme_mu': 4.5, 'extreme_sigma': 0.8,
    },
    'occlusion': {
        'phi_u': 0.96, 'sigma_eps_u': 60.0,
        'phi_v': 0.88, 'sigma_eps_v': 30.0,
    },
    'post_occlusion_phase1': {  # 1-10 frames after occlusion
        'phi_u': 0.60, 'sigma_eps_u': 25.0,
        'phi_v': 0.85, 'sigma_eps_v': 15.0,
        'mix_prob': 0.20,
        'extreme_mu': 4.8, 'extreme_sigma': 0.9,
    },
    'post_occlusion_phase2': {  # 11-30 frames after occlusion
        'phi_u': 0.88, 'sigma_eps_u': 8.0,
        'phi_v': 0.91, 'sigma_eps_v': 7.5,
        'mix_prob': 0.10,
        'extreme_mu': 4.2, 'extreme_sigma': 0.8,
    },
}

PRE_WINDOW = 30
POST_WINDOW = 30
TRANSITION_SMOOTH = 5  # frames for parameter blending at zone boundaries

# Confidence label params
CONF_PARAMS = {
    'R_high': 40.0,
    'R_decay': 90.0,
    'conf_min_high': 0.80,
    'conf_min_low': 0.10,
    'occ_conf': 0.05,
}


def read_txt(path: str) -> Optional[np.ndarray]:
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, header=None, dtype=np.float32).values
    except:
        return None


def write_txt(path: str, data: np.ndarray):
    """Write data without frame number prefix.
    uwb_obs.txt: comma-separated u,v per line
    uwb_conf.txt: single value per line
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if data.ndim == 2 and data.shape[1] == 2:
        # uwb_obs: write as "u,v"
        np.savetxt(path, data, fmt='%.6f,%.6f')
    else:
        np.savetxt(path, data, fmt='%.6f')


def find_occlusion_transitions(occ: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Find occlusion start and end indices."""
    occ_f = occ.flatten()
    diffs = np.diff(occ_f)
    starts = np.where(diffs == 1)[0] + 1
    ends = np.where(diffs == -1)[0] + 1
    return starts, ends


def assign_zones(N: int, occ: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Assign each frame a zone label:
    0=baseline, 1=pre_occlusion, 2=occlusion, 3=post_occlusion_phase1, 4=post_occlusion_phase2
    """
    zones = np.full(N, 0, dtype=np.int32)
    occ_f = occ.flatten()

    # Occlusion
    zones[occ_f == 1] = 2

    # Build nearest occlusion boundary distance
    occ_starts_set = set(starts)
    occ_ends_set = set(ends)

    for t in range(N):
        if zones[t] == 2:  # already occlusion
            continue

        # Distance to nearest occlusion start
        dist_to_start = N
        for s in starts:
            if s > t:
                dist_to_start = min(dist_to_start, s - t)
            else:
                dist_to_start = min(dist_to_start, t - s)

        # Distance to nearest occlusion end
        dist_to_end = N
        for e in ends:
            if e > t:
                dist_to_end = min(dist_to_end, e - t)
            else:
                dist_to_end = min(dist_to_end, t - e)

        min_dist = min(dist_to_start, dist_to_end)

        # Determine if pre or post occlusion
        is_pre = False
        is_post = False
        for s in starts:
            if s > t and s - t <= PRE_WINDOW:
                is_pre = True
                break
            # Also check if we're between start and end of same block
        for e in ends:
            if t >= e and t - e <= POST_WINDOW:
                is_post = True
                break

        if is_pre:
            zones[t] = 1  # pre_occlusion
        elif is_post:
            dist_from_end = t - ends[ends <= t][-1] if np.any(ends <= t) else POST_WINDOW + 1
            if dist_from_end <= 10:
                zones[t] = 3  # post_phase1
            else:
                zones[t] = 4  # post_phase2

    return zones


def generate_ar1(N: int, phi: float, sigma_eps: float, seed: Optional[int] = None) -> np.ndarray:
    """Generate AR(1) process: x[t] = φ * x[t-1] + ε[t]"""
    if seed is not None:
        np.random.seed(seed)
    x = np.zeros(N)
    if sigma_eps <= 0:
        return x
    eps = np.random.normal(0, sigma_eps, N)
    # Stationary start
    if abs(phi) < 1:
        x[0] = np.random.normal(0, sigma_eps / np.sqrt(1 - phi**2))
    else:
        x[0] = 0
    for t in range(1, N):
        x[t] = phi * x[t-1] + eps[t]
    return x


def generate_lognormal_magnitude(N: int, mu: float, sigma: float) -> np.ndarray:
    """Generate LogNormal-distributed magnitudes."""
    return np.random.lognormal(mean=mu, sigma=sigma, size=N)


def generate_isotropic_noise(magnitudes: np.ndarray) -> np.ndarray:
    """Distribute magnitude into (u,v) with random direction."""
    N = len(magnitudes)
    angles = np.random.uniform(0, 2 * np.pi, N)
    u = magnitudes * np.cos(angles)
    v = magnitudes * np.sin(angles)
    return np.column_stack([u, v])


def blend_params(phi1, sigma1, phi2, sigma2, alpha):
    """Linearly blend two sets of AR(1) parameters."""
    phi = (1 - alpha) * phi1 + alpha * phi2
    sigma = (1 - alpha) * sigma1 + alpha * sigma2
    return phi, sigma


def generate_uwb_obs(
    gt: np.ndarray,
    bbox: Optional[np.ndarray],
    occlusion: np.ndarray,
    valid: Optional[np.ndarray] = None,
    scale_baseline: float = 1.0,
    scale_pre: float = 1.0,
    scale_occ: float = 1.0,
    scale_post: float = 1.0,
    img_size: Tuple[int, int] = (1920, 1080),
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate occlusion-aware UWB observations.

    Args:
        gt: (N, 2) UWB ground truth positions
        bbox: (N, 4) or None, bounding boxes (x,y,w,h)
        occlusion: (N,) 0/1 occlusion flags
        valid: (N,) or None, 0/1 valid flags
        scale_baseline: noise scale for baseline zone
        scale_pre: noise scale for pre-occlusion zone
        scale_occ: noise scale for occlusion zone
        scale_post: noise scale for post-occlusion zones
        img_size: (width, height) for clipping
        seed: random seed

    Returns:
        obs: (N, 2) generated UWB observations
        conf: (N,) confidence values
    """
    if seed is not None:
        np.random.seed(seed)

    N = len(gt)
    occ_f = occlusion.flatten()
    starts, ends = find_occlusion_transitions(occ_f)

    # Ensure paired starts/ends
    if len(starts) == 0 and len(ends) == 0:
        occ_blocks = []
    elif len(starts) > 0 and len(ends) > 0:
        if starts[0] > ends[0]:
            starts = starts[1:] if len(starts) > 1 else np.array([], dtype=int)
            ends = ends[:len(starts)]
        min_len = min(len(starts), len(ends))
        starts, ends = starts[:min_len], ends[:min_len]
        occ_blocks = list(zip(starts, ends))
    else:
        occ_blocks = []

    # Assign zones
    zones = assign_zones(N, occ_f, starts, ends)

    # === Step 1: Baseline noise (applied everywhere as base) ===
    p = ZONE_PARAMS['baseline']
    e_base_mag = generate_lognormal_magnitude(N, p['lognorm_mu'], p['lognorm_sigma'])
    e_base = generate_isotropic_noise(e_base_mag)
    e_drift_u = generate_ar1(N, p['phi_u'], p['sigma_eps_u'])
    e_drift_v = generate_ar1(N, p['phi_v'], p['sigma_eps_v'])
    e_baseline = np.column_stack([e_base[:, 0] + e_drift_u, e_base[:, 1] + e_drift_v])
    e_baseline *= scale_baseline

    # === Step 2: Occlusion noise (replaces baseline in occ zones) ===
    p_occ = ZONE_PARAMS['occlusion']
    e_occ_drift_u = generate_ar1(N, p_occ['phi_u'], p_occ['sigma_eps_u'])
    e_occ_drift_v = generate_ar1(N, p_occ['phi_v'], p_occ['sigma_eps_v'])
    e_occlusion = np.column_stack([e_occ_drift_u, e_occ_drift_v])
    e_occlusion *= scale_occ

    # === Step 3: Extreme events (for pre-occ and post-occ transition zones) ===
    extreme = np.zeros((N, 2))

    # Pre-occlusion extreme events
    p_pre = ZONE_PARAMS['pre_occlusion']
    pre_mask = (zones == 1)
    n_pre = np.sum(pre_mask)
    if n_pre > 0:
        pre_extreme_count = int(n_pre * p_pre['mix_prob'])
        pre_extreme_idx = np.random.choice(np.where(pre_mask)[0], pre_extreme_count, replace=False)
        pre_mag = generate_lognormal_magnitude(pre_extreme_count, p_pre['extreme_mu'], p_pre['extreme_sigma'])
        extreme[pre_extreme_idx] = generate_isotropic_noise(pre_mag)
    extreme[pre_mask] *= scale_pre

    # Post-occlusion phase 1 extreme events
    p_post1 = ZONE_PARAMS['post_occlusion_phase1']
    post1_mask = (zones == 3)
    n_post1 = np.sum(post1_mask)
    if n_post1 > 0:
        post1_extreme_count = int(n_post1 * p_post1['mix_prob'])
        if post1_extreme_count > 0:
            post1_idx = np.random.choice(np.where(post1_mask)[0], post1_extreme_count, replace=False)
            post1_mag = generate_lognormal_magnitude(post1_extreme_count, p_post1['extreme_mu'], p_post1['extreme_sigma'])
            extreme[post1_idx] = generate_isotropic_noise(post1_mag)
    extreme[post1_mask] *= scale_post

    # Post-occlusion phase 2 extreme events
    p_post2 = ZONE_PARAMS['post_occlusion_phase2']
    post2_mask = (zones == 4)
    n_post2 = np.sum(post2_mask)
    if n_post2 > 0:
        post2_extreme_count = int(n_post2 * p_post2['mix_prob'])
        if post2_extreme_count > 0:
            post2_idx = np.random.choice(np.where(post2_mask)[0], post2_extreme_count, replace=False)
            post2_mag = generate_lognormal_magnitude(post2_extreme_count, p_post2['extreme_mu'], p_post2['extreme_sigma'])
            extreme[post2_idx] = generate_isotropic_noise(post2_mag)
    extreme[post2_mask] *= scale_post

    # === Step 4: Smooth transition blending at occlusion boundaries ===
    # We blend between baseline drift and occlusion drift at zone boundaries

    # Generate post-occlusion phase1 drift
    e_post1_drift_u = generate_ar1(N, p_post1['phi_u'], p_post1['sigma_eps_u'])
    e_post1_drift_v = generate_ar1(N, p_post1['phi_v'], p_post1['sigma_eps_v'])
    e_post1 = np.column_stack([e_post1_drift_u, e_post1_drift_v])

    # Generate post-occlusion phase2 drift
    e_post2_drift_u = generate_ar1(N, p_post2['phi_u'], p_post2['sigma_eps_u'])
    e_post2_drift_v = generate_ar1(N, p_post2['phi_v'], p_post2['sigma_eps_v'])
    e_post2 = np.column_stack([e_post2_drift_u, e_post2_drift_v])

    # === Step 5: Assemble final noise per zone ===
    e_total = np.copy(e_baseline)  # default for baseline zone

    # Pre-occlusion: baseline drift + extreme events
    pre_drift_scale = 1.2  # slightly elevated drift
    e_total[pre_mask] = e_baseline[pre_mask] * pre_drift_scale

    # Occlusion: pure strong drift
    occ_mask = (zones == 2)
    e_total[occ_mask] = e_occlusion[occ_mask]

    # Post-occlusion phase 1
    e_total[post1_mask] = e_post1[post1_mask] * scale_post
    # Post-occlusion phase 2
    e_total[post2_mask] = e_post2[post2_mask] * scale_post

    # Add extreme events (transition zones)
    e_total += extreme

    # === Step 6: Apply smooth transition at zone boundaries ===
    for s, e_block in occ_blocks:
        # Entry transition: 5 frames before to 5 frames after occlusion start
        for t in range(max(0, s - TRANSITION_SMOOTH), min(N, s + TRANSITION_SMOOTH)):
            alpha = (t - (s - TRANSITION_SMOOTH)) / (2 * TRANSITION_SMOOTH)
            alpha = np.clip(alpha, 0, 1)
            # Blend between baseline and occlusion parameters
            phi_u, su = blend_params(
                ZONE_PARAMS['baseline']['phi_u'], ZONE_PARAMS['baseline']['sigma_eps_u'],
                ZONE_PARAMS['occlusion']['phi_u'], ZONE_PARAMS['occlusion']['sigma_eps_u'],
                alpha
            )
            phi_v, sv = blend_params(
                ZONE_PARAMS['baseline']['phi_v'], ZONE_PARAMS['baseline']['sigma_eps_v'],
                ZONE_PARAMS['occlusion']['phi_v'], ZONE_PARAMS['occlusion']['sigma_eps_v'],
                alpha
            )
            # Use blended drift (approximate by interpolation)
            frac = alpha * scale_occ + (1 - alpha) * scale_baseline
            e_total[t] = e_total[t] * frac

        # Exit transition: 5 frames before to 5 frames after occlusion end
        for t in range(max(0, e_block - TRANSITION_SMOOTH), min(N, e_block + TRANSITION_SMOOTH)):
            alpha = (t - (e_block - TRANSITION_SMOOTH)) / (2 * TRANSITION_SMOOTH)
            alpha = np.clip(1 - alpha, 0, 1)  # reverse: occ → post
            phi_u, su = blend_params(
                ZONE_PARAMS['occlusion']['phi_u'], ZONE_PARAMS['occlusion']['sigma_eps_u'],
                ZONE_PARAMS['post_occlusion_phase1']['phi_u'], ZONE_PARAMS['post_occlusion_phase1']['sigma_eps_u'],
                alpha
            )
            phi_v, sv = blend_params(
                ZONE_PARAMS['occlusion']['phi_v'], ZONE_PARAMS['occlusion']['sigma_eps_v'],
                ZONE_PARAMS['post_occlusion_phase1']['phi_v'], ZONE_PARAMS['post_occlusion_phase1']['sigma_eps_v'],
                alpha
            )
            frac = alpha * scale_occ + (1 - alpha) * scale_post
            e_total[t] = e_total[t] * frac

    # === Step 7: Apply to GT ===
    obs = gt + e_total
    img_w, img_h = img_size
    obs[:, 0] = np.clip(obs[:, 0], 0, img_w)
    obs[:, 1] = np.clip(obs[:, 1], 0, img_h)

    # === Step 8: Generate confidence ===
    err_mag = np.sqrt(np.sum(e_total**2, axis=1))
    conf = np.ones(N) * CONF_PARAMS['conf_min_high']

    for t in range(N):
        if occ_f[t] == 1:
            conf[t] = CONF_PARAMS['occ_conf']
        elif err_mag[t] < CONF_PARAMS['R_high']:
            conf[t] = np.clip(1.0 - err_mag[t] / CONF_PARAMS['R_high'],
                              CONF_PARAMS['conf_min_high'], 1.0)
        elif err_mag[t] < CONF_PARAMS['R_decay']:
            decay = (CONF_PARAMS['R_decay'] - err_mag[t]) / (CONF_PARAMS['R_decay'] - CONF_PARAMS['R_high'])
            conf[t] = CONF_PARAMS['conf_min_high'] * decay
        else:
            conf[t] = CONF_PARAMS['conf_min_low']

    # Apply occlusion penalty to post-occ zone
    post_occ_mask = (zones == 3) | (zones == 4)
    conf[post_occ_mask] *= 0.85

    return obs, conf


def process_sequence(
    seq_dir: str,
    scale_baseline: float = 1.0,
    scale_pre: float = 1.0,
    scale_occ: float = 1.0,
    scale_post: float = 1.0,
    seed: Optional[int] = None,
    dry_run: bool = False,
):
    """Process a single sequence directory."""
    gt = read_txt(os.path.join(seq_dir, 'uwb_gt.txt'))
    occ = read_txt(os.path.join(seq_dir, 'occlusion.txt'))
    bbox = read_txt(os.path.join(seq_dir, 'groundtruth.txt'))
    valid = read_txt(os.path.join(seq_dir, 'valid.txt'))

    if gt is None or occ is None:
        print(f"  [SKIP] {seq_dir}: missing uwb_gt.txt or occlusion.txt")
        return False

    N = min(len(gt), len(occ))
    gt, occ = gt[:N], occ[:N]
    if bbox is not None:
        bbox = bbox[:N]

    # Determine image size from the first available jpg
    img_size = (1920, 1080)  # default
    for fname in sorted(os.listdir(seq_dir)):
        if fname.endswith('.jpg') or fname.endswith('.png'):
            try:
                from PIL import Image
                img = Image.open(os.path.join(seq_dir, fname))
                img_size = img.size
            except:
                pass
            break

    obs, conf = generate_uwb_obs(
        gt=gt, bbox=bbox, occlusion=occ, valid=valid,
        scale_baseline=scale_baseline,
        scale_pre=scale_pre,
        scale_occ=scale_occ,
        scale_post=scale_post,
        img_size=img_size,
        seed=seed,
    )

    if dry_run:
        print(f"  [DRY] {seq_dir}: N={N}, obs mean={np.mean(np.sqrt(np.sum((obs-gt)**2, axis=1))):.1f}px")
        return True

    # Write outputs (no frame number prefix)
    # uwb_obs.txt: comma-separated u,v
    # uwb_conf.txt: single value
    write_txt(os.path.join(seq_dir, 'uwb_obs.txt'), obs)
    write_txt(os.path.join(seq_dir, 'uwb_conf.txt'), conf.reshape(-1, 1))

    print(f"  [OK] {seq_dir}: N={N}, err_mean={np.mean(np.sqrt(np.sum((obs-gt)**2, axis=1))):.1f}px")
    return True


def process_dataset(
    root: str,
    splits: list = None,
    **kwargs,
):
    """Process all sequences in a dataset."""
    if splits is None:
        splits = ['train', 'val', 'test']

    total = 0
    success = 0
    for split in splits:
        split_dir = os.path.join(root, split)
        if not os.path.exists(split_dir):
            continue
        seq_names = sorted([d for d in os.listdir(split_dir)
                           if os.path.isdir(os.path.join(split_dir, d))])
        for seq_name in seq_names:
            seq_dir = os.path.join(split_dir, seq_name)
            total += 1
            if process_sequence(seq_dir, **kwargs):
                success += 1

    print(f"\nProcessed {success}/{total} sequences in {root}")
    return success


def parse_args():
    parser = argparse.ArgumentParser(description='Occlusion-aware UWB noise generation')
    parser.add_argument('--root', type=str, required=True,
                        help='Dataset root directory')
    parser.add_argument('--splits', type=str, nargs='+', default=['train', 'val', 'test'],
                        help='Splits to process')
    parser.add_argument('--scale-baseline', type=float, default=1.0,
                        help='Noise scale for baseline zone')
    parser.add_argument('--scale-pre', type=float, default=1.0,
                        help='Noise scale for pre-occlusion zone')
    parser.add_argument('--scale-occ', type=float, default=1.0,
                        help='Noise scale for occlusion zone')
    parser.add_argument('--scale-post', type=float, default=1.0,
                        help='Noise scale for post-occlusion zones')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    parser.add_argument('--dry-run', action='store_true',
                        help='Only print stats without writing files')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    process_dataset(
        root=args.root,
        splits=args.splits,
        scale_baseline=args.scale_baseline,
        scale_pre=args.scale_pre,
        scale_occ=args.scale_occ,
        scale_post=args.scale_post,
        seed=args.seed,
        dry_run=args.dry_run,
    )
