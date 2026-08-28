"""Shared utilities for Paper B diagnostics (B1/B2).

Reuses FastGS native components only; no training-logic modification.
"""

import os
import sys
import random

import numpy as np
import torch

# ensure repo root importable when launched as `python diagnostics/xxx.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gaussian_renderer import render_fastgs  # noqa: E402
from utils.loss_utils import l1_loss  # noqa: E402
from fused_ssim import fused_ssim as fast_ssim  # noqa: E402


# ------------------------------------------------------------------ seeds --

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------------------------------------------- checkpoint --

def clone_tree(obj):
    """Deep-copy a GaussianModel.capture() tuple.

    Adam mutates params / exp_avg in place, so every branch must restore from
    its own private copy. nn.Parameter type is preserved, otherwise restored
    tensors lose requires_grad and training silently freezes.
    """
    if torch.is_tensor(obj):
        if isinstance(obj, torch.nn.Parameter):
            return torch.nn.Parameter(obj.detach().clone(), requires_grad=obj.requires_grad)
        return obj.detach().clone()
    if isinstance(obj, dict):
        return {k: clone_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clone_tree(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(clone_tree(v) for v in obj)
    return obj


# ------------------------------------------------------- rasterizer proxy --

class CProxy:
    """Proxy for diff_gaussian_rasterization_fastgs._C recording
    num_rendered (Gaussian-tile pair count) and num_buckets per forward call.
    Zero CUDA / package modification (runtime monkeypatch only)."""

    def __init__(self, real):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "last_num_rendered", None)
        object.__setattr__(self, "last_num_buckets", None)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_real"), name)

    def rasterize_gaussians(self, *args, **kwargs):
        real = object.__getattribute__(self, "_real")
        out = real.rasterize_gaussians(*args, **kwargs)
        object.__setattr__(self, "last_num_rendered", int(out[0]))
        object.__setattr__(self, "last_num_buckets", int(out[1]))
        return out


def install_c_proxy():
    import diff_gaussian_rasterization_fastgs as pkg
    real = pkg._C
    if not isinstance(real, CProxy):
        pkg._C = CProxy(real)
    return pkg._C


# ------------------------------------------------------- native train step --

def native_train_one_iter(iteration, viewpoint_cam, gaussians, pipe, bg, opt):
    """One native FastGS training iteration (identical to train.py:78-104)."""
    gaussians.update_learning_rate(iteration)
    if iteration % 1000 == 0:
        gaussians.oneupSHdegree()
    render_pkg = render_fastgs(viewpoint_cam, gaussians, pipe, bg, opt.mult)
    image = render_pkg["render"]
    gt_image = viewpoint_cam.original_image.cuda()
    Ll1 = l1_loss(image, gt_image)
    ssim_value = fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
    loss.backward()
    return loss, render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]


# ------------------------------------------------------- projection helpers --

def project_to_pixel(cam, point_xyz):
    """Project one 3D point with cam.full_proj_transform, replicating CUDA
    transformPoint4x4 flat indexing + ndc2Pix. Returns (u, v, depth_z, valid)."""
    m = cam.full_proj_transform.flatten()
    x, y, z = float(point_xyz[0]), float(point_xyz[1]), float(point_xyz[2])
    u_h = m[0] * x + m[4] * y + m[8] * z + m[12]
    v_h = m[1] * x + m[5] * y + m[9] * z + m[13]
    w_h = m[3] * x + m[7] * y + m[11] * z + m[15]
    if abs(w_h) < 1e-9:
        return 0.0, 0.0, 0.0, False
    u_n, v_n = u_h / w_h, v_h / w_h
    w_pix, h_pix = float(cam.image_width), float(cam.image_height)
    u = ((u_n + 1.0) * w_pix - 1.0) * 0.5
    v = ((v_n + 1.0) * h_pix - 1.0) * 0.5
    return float(u), float(v), float(w_h), True


def roi_box(cx, cy, radius_px, width, height, margin_px=16):
    """Rectangular ROI = circle(cx,cy,radius) bounding box + fixed margin,
    clamped to the image. Fixed rule for every candidate/view."""
    x0 = int(max(0, np.floor(cx - radius_px - margin_px)))
    x1 = int(min(width, np.ceil(cx + radius_px + margin_px)))
    y0 = int(max(0, np.floor(cy - radius_px - margin_px)))
    y1 = int(min(height, np.ceil(cy + radius_px + margin_px)))
    return x0, y0, x1, y1


# ------------------------------------------------------------ local metrics --

def local_metrics(render_img, gt_img, box, demand_mask_full=None):
    """Local L1 / MSE / PSNR inside a fixed ROI (channel-mean convention,
    same as utils.fast_utils.get_loss). demand_mask_full: optional full-image
    bool mask; intersected with the ROI. Returns None when empty."""
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0:
        return None
    ri = render_img[:, y0:y1, x0:x1]
    gi = gt_img[:, y0:y1, x0:x1]
    if demand_mask_full is not None:
        m = demand_mask_full[y0:y1, x0:x1]
        if int(m.sum()) == 0:
            return None
        ri = ri[:, m]
        gi = gi[:, m]
    diff = (ri - gi).abs()
    l1 = float(diff.mean())
    mse = float((diff ** 2).mean())
    psnr = 10.0 * np.log10(1.0 / max(mse, 1e-10))
    return {"l1": l1, "mse": mse, "psnr": float(psnr),
            "n_pixels": int((x1 - x0) * (y1 - y0))}


def residual_descriptors(l1_map, box):
    """Residual-weighted 2D second-moment descriptors inside ROI (pre-action).

    weight = per-pixel L1 residual (channel mean). Returns energy (sum/mean),
    extent = sqrt(lam1+lam2) (px std scale), anisotropy = sqrt(lam1/lam2)
    (axis ratio), dominant direction angle (deg, principal eigenvector).
    """
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0:
        return None
    roi = l1_map[y0:y1, x0:x1]
    w = roi.flatten()
    total = float(w.sum())
    if total <= 1e-12:
        return None
    h, wdt = roi.shape
    ys, xs = torch.meshgrid(torch.arange(y0, y1, dtype=torch.float32, device=roi.device),
                            torch.arange(x0, x1, dtype=torch.float32, device=roi.device))
    xs = xs.flatten(); ys = ys.flatten()
    cx = float((w * xs).sum() / total)
    cy = float((w * ys).sum() / total)
    dx = xs - cx
    dy = ys - cy
    sxx = float((w * dx * dx).sum() / total)
    syy = float((w * dy * dy).sum() / total)
    sxy = float((w * dx * dy).sum() / total)
    cov = np.array([[sxx, sxy], [sxy, syy]], dtype=np.float64)
    evals, evecs = np.linalg.eigh(cov)  # ascending
    lam1, lam2 = float(max(evals[-1], 0.0)), float(max(evals[0], 0.0))
    aniso = float(np.sqrt(lam1 / lam2)) if lam2 > 1e-12 else float("nan")
    v1 = evecs[:, -1]
    angle = float(np.degrees(np.arctan2(v1[1], v1[0])))
    return {"residual_energy_sum": total,
            "residual_energy_mean": float(w.mean()),
            "residual_extent": float(np.sqrt(lam1 + lam2)),
            "residual_anisotropy": aniso,
            "residual_direction_deg": angle,
            "residual_centroid": [cx, cy]}


# -------------------------------------- projected Gaussian geometry (EWA) --

def projected_gaussian_geometry(cam, mean3D, scaling, rotation, radius_native=None):
    """Screen-space geometry of ONE Gaussian, replicating the rasterizer's
    computeCov3D + computeCov2D (forward.cu) in numpy.

    Returns center (u,v), 2D covariance eigen decomposition (lam1 >= lam2),
    major/minor axis angles (deg), projected anisotropy sqrt(lam1/lam2),
    extent sqrt(lam1+lam2) and the 3-sigma radius 3*sqrt(lam1) used by the
    native `radii`. If radius_native is given, asserts agreement (>=0 area
    uses ceil) and reports max mismatch for self-calibration."""
    import math
    p = np.array([float(mean3D[0]), float(mean3D[1]), float(mean3D[2])])
    s = np.array([float(scaling[0]), float(scaling[1]), float(scaling[2])])
    q = np.array([float(rotation[0]), float(rotation[1]), float(rotation[2]), float(rotation[3])])
    q = q / (np.linalg.norm(q) + 1e-12)
    r, x, y, z = q[0], q[1], q[2], q[3]
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y)],
        [2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x)],
        [2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y)]])
    Sigma = R.T @ np.diag(s * s) @ R

    vm = cam.world_view_transform.detach().cpu().numpy().flatten()  # GL row-major flat
    t = np.array([vm[0] * p[0] + vm[4] * p[1] + vm[8] * p[2] + vm[12],
                  vm[1] * p[0] + vm[5] * p[1] + vm[9] * p[2] + vm[13],
                  vm[2] * p[0] + vm[6] * p[1] + vm[10] * p[2] + vm[14]])
    if abs(t[2]) < 1e-6:
        return None
    tan_fovx = math.tan(0.5 * float(cam.FoVx))
    tan_fovy = math.tan(0.5 * float(cam.FoVy))
    W_px, H_px = float(cam.image_width), float(cam.image_height)
    fx, fy = W_px / (2 * tan_fovx), H_px / (2 * tan_fovy)
    limx, limy = 1.3 * tan_fovx, 1.3 * tan_fovy
    tx = min(limx, max(-limx, t[0] / t[2])) * t[2]
    ty = min(limy, max(-limy, t[1] / t[2])) * t[2]
    tz = t[2]
    J = np.array([
        [fx / tz, 0.0, -(fx * tx) / (tz * tz)],
        [0.0, fy / tz, -(fy * ty) / (tz * tz)],
        [0.0, 0.0, 0.0]])
    # world->view rotation matrix from the same flat viewmatrix layout
    R_w2v = np.array([
        [float(vm[0]), float(vm[4]), float(vm[8])],
        [float(vm[1]), float(vm[5]), float(vm[9])],
        [float(vm[2]), float(vm[6]), float(vm[10])]])
    cov = J @ (R_w2v @ Sigma @ R_w2v.T) @ J.T  # EWA projection (variant A)
    cov[0, 0] += 0.3
    cov[1, 1] += 0.3
    if cov[0, 0] <= 0 or cov[1, 1] <= 0 or cov[0, 0] * cov[1, 1] - cov[0, 1] ** 2 <= 0:
        return None

    # screen center (same ndc2Pix path as project_to_pixel)
    m = cam.full_proj_transform.detach().cpu().numpy().flatten()
    w_h = m[3] * p[0] + m[7] * p[1] + m[11] * p[2] + m[15]
    u_n = (m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12]) / w_h
    v_n = (m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13]) / w_h
    u = ((u_n + 1.0) * W_px - 1.0) * 0.5
    v = ((v_n + 1.0) * H_px - 1.0) * 0.5

    evals, evecs = np.linalg.eigh(cov[:2, :2])
    lam1, lam2 = float(max(evals[-1], 0.0)), float(max(evals[0], 0.0))
    major = evecs[:, -1]
    major_deg = float(np.degrees(np.arctan2(major[1], major[0]))) % 180.0
    minor_deg = (major_deg + 90.0) % 180.0
    out = {"center": [float(u), float(v)],
           "lam1": lam1, "lam2": lam2,
           "proj_anisotropy": float(np.sqrt(lam1 / lam2)) if lam2 > 1e-12 else float("nan"),
           "major_dir_deg": major_deg, "minor_dir_deg": minor_deg,
           "extent_px": float(np.sqrt(lam1 + lam2)),
           "radius_3sigma_px": float(3.0 * np.sqrt(lam1))}
    if radius_native is not None:
        out["radius_mismatch_px"] = abs(out["radius_3sigma_px"] - float(radius_native))
    return out
