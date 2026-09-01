# Copyright (c) 2026, SGA framework contributors.
"""method=exact: cat + torch.kthvalue + mask OR-merge.

The historical control path, promoted to a first-class method:
OR-merge makes the mask identical across ranks even under residual
numerical divergence — a TOLERATE-AND-CONTINUE consistency posture, one tier
above the histogram family's bitwise-replication-plus-abort.
"""

import torch


def select_masks_exact_kthvalue(grads, k, comm_group):
    """Exact topk masks: cat all scores + torch.kthvalue + mask OR-merge.

    Same threshold as the histogram path's exact tier but costs O(model)
    temporaries and a full-model uint8 mask allreduce.
    """
    all_vals = torch.cat(grads)
    abs_vals = all_vals.abs_()  # in-place; the signed views are consumed
    thresh = torch.kthvalue(abs_vals, abs_vals.numel() - k + 1).values.item()
    mask_u8 = (abs_vals >= thresh).to(torch.uint8)
    del all_vals, abs_vals

    if comm_group is not None and torch.distributed.is_initialized():
        # OR-merge so the mask is identical across ranks even under residual
        # numerical divergence (a mismatched collective would deadlock).
        # MAX of 0/1 is exactly OR and cannot wrap at any DP world size (a
        # uint8 SUM wraps mod 256 at DP >= 256, identically on all ranks —
        # invisible to every consistency layer); MAX also stays inside
        # NCCL's supported dtype/op set, unlike an int16 widening.
        torch.distributed.all_reduce(
            mask_u8, op=torch.distributed.ReduceOp.MAX, group=comm_group
        )

    masks = []
    offset = 0
    for g in grads:
        n = g.numel()
        masks.append(mask_u8[offset:offset + n])
        offset += n
    return thresh, masks


class ExactKthvalue:
    """Impl signature: ``select(grads, k, comm_group, warmup_target)``."""

    def select(self, grads, k, comm_group, warmup_target=None):
        return select_masks_exact_kthvalue(grads, k, comm_group)
