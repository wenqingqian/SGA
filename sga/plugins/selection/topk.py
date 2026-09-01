# Copyright (c) 2026, SGA framework contributors.
"""Topk selection — the selection axis' single registration subject.

Strict-ratio global topk over the optimizer-materialized score magnitudes:
``k = max(1, round(n_total * ratio%))``. The selection ALGORITHM is the
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
a loud refusal). Historical note: this subject was previously registered as
``circular_topk``.

"Circular" (the support is regenerated once per cycle period T at the
full-sync boundary and persists unchanged in between; non-topk coordinates
accumulate as residual, folded by the optimizer's reset contract at the
next boundary) is ENGINE-CYCLE semantics, not a property of this selector —
hence the bare registration name ``topk``.
"""

from sga.plugins import (
    PluginParam,
    SelectionDecl,
    resolve_plugin_params,
)
from sga.plugins.selection._adaptive import AdaptiveTau
from sga.plugins.selection._exact import ExactKthvalue
from sga.plugins.selection._histogram import _HIST_MODES, HistogramSelect

_DECL_PARAMS = (
    PluginParam(
        "method", str, required=True,
        choices=("adaptive", "exact", "histogram"),
        doc="selection algorithm: adaptive tau controller / bit-pattern "
            "histogram / kthvalue+OR-merge",
    ),
    PluginParam(
        "ratio_percent", float, required=True,
        doc="topk mask ratio in percent, strictly inside (0, 100)",
    ),
    PluginParam(
        "histogram_mode", str, default="exact",
        choices=tuple(sorted(_HIST_MODES)),
        doc="method=histogram: geometry tier exact / balanced / fast",
    ),
    PluginParam(
        "adaptive_warmup_syncs", int, default=5, nullable=True,
        doc="method=adaptive: gated warmup selects before the persistent "
            "tau takes over (null disables the warmup gate)",
    ),
    PluginParam(
        "adaptive_warmup_mode", str, default="fast",
        choices=tuple(sorted(_HIST_MODES)),
        doc="method=adaptive: histogram tier used to bootstrap/steer the "
            "adaptive tau",
    ),
)

def effective_params(params: dict) -> dict:
    """Method-aware view of a RESOLVED param dict for the resolve CLI: keep
    only the keys the configured method consumes. The union declaration's
    other defaults would be foreign keys — the dump's own output would be
    rejected by the constructor's per-method check."""
    keep = ("method", "ratio_percent", *_METHOD_KEYS[params["method"]])
    return {k: v for k, v in params.items() if k in keep}


DECL = SelectionDecl(name="topk", output_form="mask", params=_DECL_PARAMS,
                     effective=effective_params)

# Per-method vocabularies (the declaration carries the union — the flat
# flag table must offer every param — while a CONFIG may only use the keys
# of its own method).
_METHOD_KEYS = {
    "adaptive": ("adaptive_warmup_syncs", "adaptive_warmup_mode"),
    "histogram": ("histogram_mode",),
    "exact": (),
}


# Every method-specific key is foreign to SOME method; membership here is
# what makes the per-method rejection below a closed check. A param added to
# the declaration but forgotten in _METHOD_KEYS would silently become
# accepted-under-every-method — the exact failure mode that check exists to
# prevent, so the sync is enforced at import.
_FOREIGN_POSSIBLE = frozenset().union(*_METHOD_KEYS.values())
assert _FOREIGN_POSSIBLE == {p.name for p in _DECL_PARAMS} - {"method", "ratio_percent"}, (
    "topk DECL params and _METHOD_KEYS are out of sync"
)


class TopkSelection:

    decl = DECL

    def __init__(self, params: dict):
        raw = dict(params)  # pre-resolution view (explicit keys only)
        method = raw.get("method")
        # Foreign-key check FIRST: a wrong-method knob is a vocabulary error
        # and must not surface as (or be masked by) a type/choices complaint.
        if method in _METHOD_KEYS:
            foreign = sorted(
                key for key in raw
                if key in _FOREIGN_POSSIBLE and key not in _METHOD_KEYS[method]
            )
            if foreign:
                own = sorted(_METHOD_KEYS[method]) or "no extra params"
                raise ValueError(
                    f"sga.selection (topk): params {foreign} do not belong to "
                    f"method '{method}' (it takes {own}) — under this method "
                    "they would be silently dead; drop them or switch the "
                    "method"
                )
        params = resolve_plugin_params("selection", "topk", DECL, raw)
        self.params = params
        self.method = method
        self.ratio_percent = float(params["ratio_percent"])
        assert 0.0 < self.ratio_percent < 100.0, (
            f"ratio_percent must be in (0, 100), got {self.ratio_percent}"
        )
        if method == "adaptive":
            warmup_syncs = params["adaptive_warmup_syncs"]
            self._impl = AdaptiveTau(
                self.ratio_percent,
                int(warmup_syncs) if warmup_syncs is not None else None,
                params["adaptive_warmup_mode"],
            )
        elif method == "histogram":
            self._impl = HistogramSelect(params["histogram_mode"])
        else:
            self._impl = ExactKthvalue()
        self.last_threshold = None

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

