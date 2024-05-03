import torch
from scipy.spatial import cKDTree

def closest_point_distance(points, other_points):
    tree = cKDTree(other_points)
    dists, _ = tree.query(points, k=1)
    return torch.from_numpy(dists)  # Convert back to Torch tensor for further computations

def nn_distance(pc1, pc2):
    """
    Input:
        pc1, pc2: (B,N,C) and (B,M,C) torch tensors
    Output:
        dist1, idx1, dist2, idx2
    """
    batch_size = pc1.shape[0]
    dist1 = torch.empty(pc1.shape[:2])
    dist2 = torch.empty(pc2.shape[:2])

    # Compute for each batch
    for i in range(batch_size):
        pc1_np = pc1[i].cpu().numpy()  # Convert to NumPy for cKDTree
        pc2_np = pc2[i].cpu().numpy()

        dist1[i] = closest_point_distance(pc1_np, pc2_np)
        dist2[i] = closest_point_distance(pc2_np, pc1_np)

    # Dummy indices (not used but maintained for compatibility)
    idx1 = torch.zeros_like(dist1, dtype=torch.int64)
    idx2 = torch.zeros_like(dist2, dtype=torch.int64)

    return dist1, idx1, dist2, idx2

def chamfer_raw(pc_a, pc_b, swap_axes=False):
    """ Compute the chamfer loss for batched pointclouds.
    :param pc_a: torch.Tensor B x Na-points per point-cloud x 3
    :param pc_b: torch.Tensor B x Nb-points per point-cloud x 3
    :return: dist_a: torch.Tensor, dist_b: torch.Tensor
    # Note: this is 10x slower than the chamfer_loss in losses/chamfer.py BUT this plays also in CPU (the
    other does not).
    """
    if swap_axes:
        pc_a = pc_a.transpose(-1, -2).contiguous()
        pc_b = pc_b.transpose(-1, -2).contiguous()
    dist_a, _, dist_b, _ = nn_distance(pc_a, pc_b)

    return dist_a, dist_b