"""
Pure-PyTorch implementations of pointnet2 operations.
Drop-in replacement for pointnet2._ext CUDA extension (inference only).
All functions match the C-extension signatures exactly.
"""
import torch


# ── furthest_point_sampling ──
def furthest_point_sampling(xyz, npoint):
    """(B, N, 3) -> (B, npoint) indices"""
    B, N, _ = xyz.shape
    device = xyz.device
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids


# ── gather_points ──
def gather_points(features, idx):
    """(B, C, N), (B, npoint) or (B, npoint, nsample) -> (B, C, npoint) or (B, C, npoint, nsample)"""
    B = features.shape[0]
    C = features.shape[1]
    if idx.dim() == 2:
        # (B, C, N), (B, npoint) -> (B, C, npoint)
        idx_expanded = idx.unsqueeze(1).expand(-1, C, -1)
        return torch.gather(features, 2, idx_expanded)
    else:
        # (B, C, N), (B, npoint, nsample) -> (B, C, npoint, nsample)
        npoint, nsample = idx.shape[1], idx.shape[2]
        idx_expanded = idx.unsqueeze(1).expand(-1, C, -1, -1)
        return torch.gather(features.unsqueeze(2).expand(-1, -1, npoint, -1), 3, idx_expanded)


# ── gather_points_grad ──
def gather_points_grad(grad_out, idx, N):
    """(B, C, npoint, nsample), (B, npoint, nsample), N -> (B, C, N)"""
    B, C = grad_out.shape[0], grad_out.shape[1]
    grad_features = torch.zeros(B, C, N, device=grad_out.device, dtype=grad_out.dtype)
    grad_features.scatter_add_(2, idx.unsqueeze(1).expand(-1, C, -1, -1).reshape(B, C, -1),
                               grad_out.reshape(B, C, -1))
    return grad_features


# ── three_nn ──
def three_nn(unknown, known):
    """(B, N, 3), (B, M, 3) -> (dist2, idx)"""
    dist2 = ((unknown.unsqueeze(2) - known.unsqueeze(1)) ** 2).sum(-1)
    dist2, idx = torch.topk(dist2, k=3, dim=-1, largest=False)
    return dist2, idx.int()


# ── three_interpolate ──
def three_interpolate(features, idx, weight):
    """(B, C, M), (B, N, 3), (B, N, 3) -> (B, C, N)"""
    B, C, M = features.shape
    N = idx.shape[1]
    # Gather: idx is (B, N, 3), features is (B, C, M)
    # Result should be (B, C, N, 3) — one feature vector per neighbor
    idx_expanded = idx.long().unsqueeze(1).expand(-1, C, -1, -1)  # (B, C, N, 3)
    gathered = torch.gather(features.unsqueeze(2).expand(-1, -1, N, -1), 3, idx_expanded)  # (B, C, N, 3)
    return (gathered * weight.unsqueeze(1)).sum(-1)  # (B, C, N)


# ── three_interpolate_grad ──
def three_interpolate_grad(grad_out, idx, weight, M):
    """(B, C, N), (B, N, 3), (B, N, 3), M -> (B, C, M)"""
    B, C = grad_out.shape[0], grad_out.shape[1]
    N = idx.shape[1]
    grad_features = torch.zeros(B, C, M, device=grad_out.device, dtype=grad_out.dtype)
    grad_out_w = grad_out.unsqueeze(2) * weight.unsqueeze(1)  # (B, C, N, 3)
    idx_expanded = idx.unsqueeze(1).expand(-1, C, -1, -1)  # (B, C, N, 3)
    grad_features.scatter_add_(2, idx_expanded.reshape(B, C, -1), grad_out_w.reshape(B, C, -1))
    return grad_features


# ── group_points ──
def group_points(features, idx):
    """(B, C, N), (B, npoint, nsample) -> (B, C, npoint, nsample)"""
    B, C = features.shape[0], features.shape[1]
    npoint, nsample = idx.shape[1], idx.shape[2]
    idx_flat = idx.reshape(B, -1)
    gathered = features[batch_indices_2d(B, npoint * nsample, features.device).unsqueeze(1), :, idx_flat]
    return gathered.reshape(B, C, npoint, nsample)


def batch_indices_2d(B, N, device):
    return torch.arange(B, device=device).unsqueeze(1).expand(B, N)


# ── group_points_grad ──
def group_points_grad(grad_out, idx, N):
    """(B, C, npoint, nsample) -> (B, C, N)"""
    B, C = grad_out.shape[0], grad_out.shape[1]
    grad_features = torch.zeros(B, C, N, device=grad_out.device, dtype=grad_out.dtype)
    idx_flat = idx.reshape(B, -1).unsqueeze(1).expand(-1, C, -1)
    grad_flat = grad_out.reshape(B, C, -1)
    grad_features.scatter_add_(2, idx_flat, grad_flat)
    return grad_features


# ── ball_query ──
def ball_query(new_xyz, xyz, radius, nsample):
    """(B, M, 3), (B, N, 3) -> (B, M, nsample) idx"""
    new_xyz = new_xyz.contiguous()
    while new_xyz.dim() > 3:
        new_xyz = new_xyz.squeeze(2)
    B, M, C = new_xyz.shape
    N = xyz.shape[1]
    assert C == 3, f"Expected 3 channels, got {C} {new_xyz.shape}"
    # Use torch.cdist for numerical stability and speed
    k = min(nsample, N)
    dist2 = torch.cdist(new_xyz, xyz)  # (B, M, N)
    _, idx = torch.topk(dist2, k=k, dim=-1, largest=False)
    # Ensure idx is within [0, N-1]
    return idx.clamp(0, N - 1).int()


# ── cylinder_query ──
def cylinder_query(new_xyz, xyz, rot, radius, hmin, hmax, nsample):
    """(B, M, 3), (B, N, 3), (B, M, 9) -> (B, M, nsample) idx"""
    while new_xyz.dim() > 3:
        new_xyz = new_xyz.squeeze(2)
    B, M, _ = new_xyz.shape
    N = xyz.shape[1]
    # rot is (B, M, 9) — reshape to (B, M, 3, 3)
    rot_mat = rot.reshape(B, M, 3, 3)
    delta = xyz.unsqueeze(1) - new_xyz.unsqueeze(2)  # (B, M, N, 3)
    # Apply rotation: (B, M, 3, 3) @ (B, M, N, 3, 1) -> (B, M, N, 3, 1)
    xyz_local = (rot_mat.unsqueeze(2) @ delta.unsqueeze(-1)).squeeze(-1)
    r2 = xyz_local[..., 0] ** 2 + xyz_local[..., 1] ** 2
    h = xyz_local[..., 2]
    mask = (r2 <= radius ** 2) & (h >= hmin) & (h <= hmax)
    score = torch.where(mask, r2 + (h - (hmin + hmax) / 2) ** 2,
                        torch.tensor(1e10, device=xyz.device))
    _, idx = torch.topk(-score, k=min(nsample, score.shape[-1]), dim=-1)
    return idx.int()
