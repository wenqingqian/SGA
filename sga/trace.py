# Copyright (c) 2026, SGA framework contributors.
"""Engine-side pre-commit trace collector (offline replay data source).

Dumps the optimizer's fp32 main-param view right before the AdamW commit:
per-step gradients ``g`` (post unscale + clip) inside configured windows,
and full ``w/m/v`` state at branch points. Consumed offline by
``python -m sga.replay`` to evaluate oracle score candidates against future
update distortion without touching training semantics. Read-only: the
collector never mutates any tensor consumed by training (synchronous
``.cpu()`` copies only, world-rank-0 only — deliberately NOT async: the dump
runs at most once per step inside a configured window and a silent overlap
hazard costs more than the copy).

Format is schema-compatible with the legacy in-fork collector (catalog.pt /
step_NNNNNN_state.pt / step_NNNNNN_g.pt): mutually readable, with two
ADDITIVE extensions the legacy replay simply ignores — catalog entries
carry ``param_group`` and the meta carries ``param_groups`` (per-group
lr/betas/eps/wd), letting the new replay resolve exact per-group
hyperparameters. Catalog keys are canonical
parameter names with every leading ``module.`` wrapper segment stripped —
the legacy dumps keyed by ``module.module.*``, which tied traces to the
wrapping stack for no benefit. The legacy ``_sga_trace_name`` param tagging
and ``_sga_last_grad_norm`` staging are gone: the engine derives names from
the attached chunks and reads the grad norm from its own clip.

Numbering is the fork's ABSOLUTE iteration (0-indexed at the optimizer-step
call point, matching the legacy local counter for runs starting from
scratch: dump at step t is the pre-commit state with exactly t commits
applied, and g at step t is consumed by commit t+1). Absolute numbering
fixes the legacy resume-overwrite hazard (a resumed run's local counter
restarted at 0 and silently overwrote early dumps; now a resumed run's
dumps continue at its resume iteration and never collide). A found-inf
skipped step leaves a GAP in the g-file sequence; the replay's contiguous
window discovery stops there instead of silently renumbering.

Dump slot (frozen ordering contract): in ``SGAEngine.optimizer_step``
AFTER step 3 (count_zeros) and BEFORE step 4 (the optimizer template owns
the inner step
and rewrites main_param.grad). At that point prepare_grads + the masked
clip have run, so main_param.grad is the post-unscale post-clip gradient —
exactly the legacy hook's view at the top of step_with_ready_grads. On
sparse steps that gradient holds reduced values only at support coords
(masked clip semantics); replay-grade traces come from full/dense (shadow)
runs where every coord is reduced — the format does not distinguish.
Compact-emission sparse steps: the values are scattered back
into the full layout for the dump window and re-zeroed right after, so
the dump reads the same view the writeback path produced; outside the
dump window the unselected coords hold the raw accumulated residual
(pack clears only the communicated positions) — same semantic as the
writeback path, nothing dump-visible changes.

Enabled by ``sga_trace_dir`` (yaml ``sga:`` section); absent (default)
disables collection entirely: the engine holds ``_trace = None`` and the
slot costs one None-check per step (zero-overhead-when-off, asserted
structurally in tests; the ≤0.5% step-time gate is measured in the
single-node validation).
"""

import os
from typing import Dict, List, Optional

import torch


def parse_branch_points(spec) -> List[int]:
    """Normalize a branch-point spec (comma string or int list) into a
    sorted list of unique non-negative ints. Empty/None -> []. Non-integral
    floats are rejected: int() would silently truncate 10.5 to 10 and land
    the window on the wrong step."""
    if spec is None:
        return []
    if isinstance(spec, str):
        toks = [t for t in spec.replace(" ", "").split(",") if t]
    else:
        toks = list(spec)
    points = set()
    for tok in toks:
        if isinstance(tok, float) and not tok.is_integer():
            raise ValueError(
                f"trace branch point must be an integer, got {tok!r}"
            )
        try:
            t = int(tok)
        except (TypeError, ValueError):
            raise ValueError(
                f"trace branch point must be an integer, got {tok!r}"
            ) from None
        if t < 0:
            raise ValueError(f"trace branch point must be >= 0, got {t}")
        points.add(t)
    return sorted(points)


def _canonical_name(name: str) -> str:
    """Strip every leading ``module.`` wrapper segment (DDP / Float16Module)
    so catalog keys do not depend on the wrapping stack."""
    while name.startswith("module."):
        name = name[len("module."):]
    return name


def _dump_phase(engine) -> str:
    """Engine mode at dump time: shadow, or the live step's sync phase."""
    if getattr(engine, "_shadow", False):
        return "shadow"
    return "live_full" if getattr(engine, "_step_full", None) else "live_sparse"


def _catalog_mismatch(old, expected: dict):
    """First offending (name, field) between an on-disk catalog and the one
    this run would write — key set plus shape/dtype/param_group per entry —
    or None when identical. The resume identity anchor: a catalog from a
    different model/config would otherwise silently combine with new step
    payloads and poison replay."""
    old = old if isinstance(old, dict) else {}
    if set(old) != set(expected):
        for name in expected:
            if name not in old:
                return name, "name"
        for name in old:
            if name not in expected:
                return name, "name"
    for name, ent in expected.items():
        for field in ("shape", "dtype", "param_group"):
            if old[name].get(field) != ent[field]:
                return name, field
    return None


class TraceCollector:
    """Per-engine trace state; constructed only when sga_trace_dir is set.

    The engine passes the canonical values of ``sga_trace_dir`` /
    ``sga_trace_branch_points`` / ``sga_trace_window`` straight through
    (already typed and validated; ``parse_branch_points`` below is the
    idempotent guard for direct construction)."""

    def __init__(self, dir: str, branch_points, window: int):
        self.dir = dir
        self.branch_points = parse_branch_points(branch_points)
        self.window = int(window)
        self._entries: Optional[list] = None   # [(name, rec, group_index)]
        self._catalog_written = False          # per process (resume-safe)

    def wants(self, step: int):
        """(want_state, want_g) for an absolute engine step."""
        want_state = step in self.branch_points
        want_g = want_state or any(
            t0 <= step < t0 + self.window for t0 in self.branch_points
        )
        return want_state, want_g

    # ------------------------------------------------------------------ bind

    def _bind(self, engine, mp_opt) -> None:
        """Resolve (canonical name, record, optimizer group) per record once.

        Names come from the attached chunks' ``module.named_parameters()``
        (no param tagging); the group index from the inner optimizer's
        param_groups. Both maps are identity-keyed and stable for the run.
        """
        name_of: Dict[int, str] = {}
        for chunk in engine._chunks.values():
            for n, p in chunk.module.named_parameters():
                name_of[id(p)] = _canonical_name(n)
        group_of: Dict[int, int] = {}
        for gi, group in enumerate(mp_opt.optimizer.param_groups):
            for p in group["params"]:
                if id(p) in group_of and group_of[id(p)] != gi:
                    raise ValueError(
                        "trace: main param appears in multiple optimizer "
                        f"param_groups ({group_of[id(p)]} and {gi})"
                    )
                group_of[id(p)] = gi
        entries = []
        seen = set()
        for rec in engine._records:
            name = name_of.get(id(rec.model_param))
            if name is None:
                raise ValueError(
                    "trace: model param not named by any attached chunk "
                    f"(shape {tuple(rec.model_param.shape)}); attach() must "
                    "precede the first dump"
                )
            if name in seen:
                raise ValueError(
                    f"trace: duplicate canonical param name '{name}' after "
                    "module.-prefix stripping"
                )
            seen.add(name)
            gi = group_of.get(id(rec.main_param))
            if gi is None:
                raise ValueError(
                    "trace: main param not found in any optimizer param_group "
                    f"(model param shape {tuple(rec.model_param.shape)})"
                )
            entries.append((name, rec, gi))
        self._entries = entries

    # ------------------------------------------------------------------ dump

    def _meta(self, mp_opt, step: int, grad_norm, phase: Optional[str]) -> dict:
        cfg = mp_opt.config
        groups = []
        for g in mp_opt.optimizer.param_groups:
            betas = g.get("betas", (0.9, 0.999))
            groups.append({
                "lr": float(g.get("lr", 0.0)),
                "weight_decay": float(g.get("weight_decay", 0.0)),
                "adam_beta1": float(betas[0]),
                "adam_beta2": float(betas[1]),
                "adam_eps": float(g.get("eps", 1e-8)),
            })
        meta = {
            "step": step,
            # Engine mode at dump time: 'shadow' | 'live_full' | 'live_sparse'.
            # Replay-grade traces come from shadow/dense runs; the field makes
            # a mixed directory self-describing instead of inferrable.
            "phase": phase,
            "lr": [g["lr"] for g in groups],
            "grad_norm": float(grad_norm) if grad_norm is not None else float("nan"),
            "clip_grad": float(getattr(cfg, "clip_grad", 0.0)),
            # legacy global scalars (uniform-run replay fallback; per-group
            # values take precedence when the catalog carries param_group)
            "weight_decay": float(getattr(cfg, "weight_decay", 0.0)),
            "adam_beta1": float(getattr(cfg, "adam_beta1", 0.9)),
            "adam_beta2": float(getattr(cfg, "adam_beta2", 0.999)),
            "adam_eps": float(getattr(cfg, "adam_eps", 1e-8)),
            "param_groups": groups,
        }
        return meta

    @torch.no_grad()
    def maybe_dump(self, engine, mp_opt, step: int, grad_norm) -> None:
        """Dump slot: called once per optimizer_step when tracing is on."""
        want_state, want_g = self.wants(step)
        if not (want_state or want_g):
            return
        if engine._rank != 0:
            return
        records = engine._records or engine._refresh_records(mp_opt)
        if self._entries is None:
            self._bind(engine, mp_opt)

        os.makedirs(self.dir, exist_ok=True)
        meta = self._meta(mp_opt, step, grad_norm, _dump_phase(engine))

        if not self._catalog_written:
            cat_path = os.path.join(self.dir, "catalog.pt")
            catalog = {
                name: {"shape": list(rec.main_param.shape),
                       "dtype": str(rec.main_param.dtype),
                       "param_group": gi}
                for name, rec, gi in self._entries
            }
            if os.path.exists(cat_path):
                # Resume: verify the producing run's identity, never rewrite.
                # With the catalog verified, re-dumping the same absolute
                # steps is a legitimate same-step recomputation after a
                # mid-window crash, so existing step files stay unguarded.
                on_disk = torch.load(cat_path, weights_only=False)
                old = on_disk.get("catalog") if isinstance(on_disk, dict) else None
                mismatch = _catalog_mismatch(old, catalog)
                if mismatch is not None:
                    raise ValueError(
                        "trace: existing catalog.pt in "
                        f"{self.dir} does not match this run "
                        f"({mismatch[1]} for {mismatch[0]})"
                    )
            else:
                torch.save({"meta": meta, "catalog": catalog}, cat_path)
            self._catalog_written = True

        if want_state:
            self._dump_state(step, meta)
        if want_g:
            self._dump_grad(step, meta)

    def _dump_state(self, step: int, meta: dict) -> None:
        """Dump w/m/v (+ g) at a branch point. Missing moments (lazy AdamW
        state before the first commit) are dumped as zeros and flagged."""
        state_absent = False
        payload = {"w": {}, "m": {}, "v": {}, "g": {}}
        for name, rec, _gi in self._entries:
            payload["w"][name] = rec.main_param.data.detach().cpu()
            if rec.main_param.grad is not None:
                payload["g"][name] = rec.main_param.grad.detach().cpu()
            st = rec._state or {}
            if "exp_avg" in st and "exp_avg_sq" in st:
                payload["m"][name] = st["exp_avg"].detach().cpu()
                payload["v"][name] = st["exp_avg_sq"].detach().cpu()
            else:
                state_absent = True
                payload["m"][name] = torch.zeros_like(payload["w"][name])
                payload["v"][name] = torch.zeros_like(payload["w"][name])
        step_meta = dict(meta)
        step_meta["state_absent"] = state_absent
        torch.save({"meta": step_meta, "payload": payload},
                   os.path.join(self.dir, f"step_{step:06d}_state.pt"))

    def _dump_grad(self, step: int, meta: dict) -> None:
        payload = {}
        for name, rec, _gi in self._entries:
            if rec.main_param.grad is not None:
                payload[name] = rec.main_param.grad.detach().cpu()
        torch.save({"meta": dict(meta), "payload": payload},
                   os.path.join(self.dir, f"step_{step:06d}_g.pt"))
