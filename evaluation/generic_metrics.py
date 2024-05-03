import torch
import sys
sys.path.append('../changeit3d')
from changeit3d.utils.basics import iterate_in_chunks
from losses.chamfer import chamfer_loss


def chamfer_dists(original_shapes, transformed_shapes, bsize, device="cuda"):
    n_items = len(original_shapes)
    all_dists = []
    assert n_items == len(transformed_shapes)
    for locs in iterate_in_chunks(range(n_items), bsize):
        locs_list = list(locs)

        o = torch.Tensor([original_shapes[i] for i in locs_list]).to(device)
        t = torch.Tensor([transformed_shapes[i] for i in locs_list]).to(device)
       
        chamfer, _ = chamfer_loss(o, t, swap_axes=False, reduction=None)
        all_dists.append(chamfer)
    all_dists = torch.cat(all_dists)

    return torch.mean(all_dists).item(), all_dists