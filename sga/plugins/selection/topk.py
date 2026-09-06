# Copyright (c) 2026, SGA framework contributors.
"""Topk selection — the selection axis' single registration subject.

Strict-ratio global topk over the optimizer-materialized score magnitudes:
``k = max(1, round(n_total * ratio%))``. Under ``sga_always_sync_layers``
the pool is the engine-filtered eligible subset — forced params never
compete, and ``n_total`` (like every ratio statistic) is the eligible
count (see ``SelectionContext``). The selection ALGORITHM is the
``method`` parameter, dispatched to a per-method impl file:

- ``method: histogram`` (+ ``histogram_mode: exact|balanced|fast``):
  bit-pattern histogram select (``_histogram.py``). Per-rank histograms of
  the top bits of |score| fp32 bit patterns are SUM-allreduced over the comm
  group; with replicated inputs the merged histogram is exactly ``world``
  times the local one and the select math is scale-invariant, so every rank
  derives the identical bin and threshold BY CONSTRUCTION (tied boundary
  values are all selected — masks ``|score| >= thresh``). The default
  (``exact`` mode) does NOT OR-merge the final masks: it leans on
  bitwise-replicated inputs plus the engine's layer-2 abort on divergence —
  merging there too would trade a crash for consistency at the cost of an
  extra uint8 allreduce per reselection, which v1 declines deliberately.
- ``method: exact`` (``_exact.py``): full-model cat + torch.kthvalue +
  mask OR-merge. One consistency tier above the histogram family: OR-merge
  TOLERATES residual cross-rank numerical divergence instead of aborting
  (costs O(model) temporaries + a full-model uint8 mask allreduce).
- ``method: adaptive`` (+ ``adaptive_warmup_syncs``/``adaptive_warmup_mode``,
  ``_adaptive.py``): persistent-tau controller — gated histogram warmup,
  then log-space dead-band feedback toward the target ratio.

Params of one method are foreign keys under another and rejected at build
time (their knobs have no effect there — a silently-dead knob is worse than
a loud refusal); the per-method vocabularies are this plugin's own (the
literal UNLOCKS table in this module mirrors the config-side rule).
Historical note: this subject was previously registered as
``circular_topk``.

"Circular" (the support is regenerated once per cycle period T at the
full-sync boundary and persists unchanged in between; non-topk coordinates
accumulate as residual, folded by the optimizer's reset contract at the
next boundary) is ENGINE-CYCLE semantics, not a property of this selector —
hence the bare registration name ``topk``.
"""

from sga.config.validation import declared_choices
from sga.plugins import SelectionDecl, expand_plugin_params
from sga.plugins.selection._adaptive import AdaptiveTau
from sga.plugins.selection._exact import ExactKthvalue
from sga.plugins.selection._histogram import HistogramSelect

DECL = SelectionDecl(name="topk", output_form="mask")

# topk's own vocabulary: its params as FULL flat dests (defaults and
# required-ness live in the argparse declarations — a None default means
# required); UNLOCKS says which method each guarded param belongs to.
PARAMS = ("sga_selection_method", "sga_selection_ratio_percent",
          "sga_selection_histogram_mode",
          "sga_selection_adaptive_warmup_syncs",
          "sga_selection_adaptive_warmup_mode",
          "sga_selection_block_size", "sga_selection_block_reduce")
UNLOCKS = {"histogram": ("sga_selection_histogram_mode",
                         "sga_selection_block_size",
                         "sga_selection_block_reduce"),
           "adaptive": ("sga_selection_adaptive_warmup_syncs",
                        "sga_selection_adaptive_warmup_mode")}


class TopkSelection:

    decl = DECL

    def __init__(self, params: dict):
        raw = dict(params)  # pre-expansion view (explicit keys only)
        method = raw.get("sga_selection_method")
        # Dead-config check FIRST, gated on a KNOWN method: a missing or
        # invalid method must surface as the shared validator's
        # required/choices error, not as a dead-config refusal (which would
        # name the raw value and mask the real problem).
        methods = declared_choices("sga_selection_method")
        if method in methods:
            own = UNLOCKS.get(method, ())
            dead = sorted(p for ps in UNLOCKS.values() for p in ps
                          if p in raw and p not in own)
            if dead:
                raise ValueError(
                    f"sga_selection (topk): params {dead} do not belong to "
                    f"method '{method}' (it takes {sorted(own) or 'no extra params'}) — "
                    "under this method they would be silently dead; drop them "
                    "or switch the method"
                )
        params = expand_plugin_params(PARAMS, raw, "sga_selection (topk)")
        self.params = params
        self.method = method
        self.ratio_percent = float(params["sga_selection_ratio_percent"])
        assert 0.0 < self.ratio_percent < 100.0, (
            f"ratio_percent must be in (0, 100), got {self.ratio_percent}"
        )
        if method == "adaptive":
            warmup_syncs = params["sga_selection_adaptive_warmup_syncs"]
            self._impl = AdaptiveTau(
                self.ratio_percent,
                int(warmup_syncs) if warmup_syncs is not None else None,
                params["sga_selection_adaptive_warmup_mode"],
            )
        elif method == "histogram":
            self._impl = HistogramSelect(
                params["sga_selection_histogram_mode"],
                block_size=int(params["sga_selection_block_size"]),
                block_reduce=params["sga_selection_block_reduce"])
        else:
            self._impl = ExactKthvalue()
        self.last_threshold = None

    @property
    def block_size(self):
        """Selection granularity: 1 = element-level (every non-histogram
        method); > 1 = each selected coordinate heads a contiguous run of
        ``block_size`` elements. The engine reads this to decide whether a
        block-index form exists for the support."""
        return getattr(self._impl, "block_size", 1)

    @property
    def tau(self):
        """Adaptive impl's persistent threshold (None for other methods)."""
        return getattr(self._impl, "tau", None)

    @property
    def warmup_passed(self):
        """Adaptive impl's consecutive-gate count (None for other methods)."""
        return getattr(self._impl, "warmup_passed", None)

    def k_for(self, n_total: int) -> int:
        return max(1, int(round(n_total * self.ratio_percent / 100.0)))

    def warmup_sync_target(self, start_iter: int, period: int):
        """Engine hook (adaptive-only): gated warmup selects implied by this
        selection's warmup window, anchored at start_iter. None for every
        other method and for adaptive-with-warmup-disabled."""
        hook = getattr(self._impl, "warmup_sync_target", None)
        return hook(start_iter, period) if callable(hook) else None

    def select(self, ctx):
        """Return per-bucket element-level uint8 masks from the global topk.

        ``ctx.grad_views`` are the optimizer-materialized score views;
        selection runs only at full-sync cycle boundaries.
        """
        k = self.k_for(sum(g.numel() for g in ctx.grad_views))
        thresh, masks = self._impl.select(
            ctx.grad_views, k, ctx.comm_group,
            getattr(ctx, "warmup_target", None),
        )
        self.last_threshold = thresh
        return masks

