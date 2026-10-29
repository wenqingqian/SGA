# Copyright (c) 2026, SGA framework contributors.
"""SGA offline replay oracle: score evaluation on recorded training traces.

consumes pre-commit trace dumps produced by the engine-side trace
collector (sga.trace config) and,
offline, computes per-coordinate oracle scores and evaluates their
predictive power over a replay window. Trace format is byte-compatible with
the legacy collector (catalog.pt / step_NNNNNN_state.pt / step_NNNNNN_g.pt),
so this tool replays legacy traces and the legacy tool replays new ones.

Scores come from the framework's score REGISTRY (sga/plugins/scores.py) —
the same names and formulas the online engine and shadow mode use; this tool
consumes the registry directly and defines no score set of its own:
    abs_g        = |g|                       (observed, post-clip gradient)
    abs_m        = |m|                       (pre-commit first moment)
    update       = |wd*w + U(g)|             (online full-sync selection score,
                                              upcoming-commit bias corrections)
    coi          = |U(g) - U(0)|             (counterfactual optimizer impact)
(fp64 arithmetic, fp32 storage; the materialized form may be signed — the
ranking selects store magnitudes.)

Replay modes:
- ``zerofill``: frozen support, zero-filled q each step (original offline
  semantics), replayed in-process;
- ``cycle``: residual accumulation + boundary full-sync fold + mask
  re-selection every --cycle-len steps (offline twin of the online cyclic
  semantics), always served by the fused single-pass engine
  (sga/replay_fused.py) and therefore requiring --dense-cache (populated
  automatically on the first run). Boundary steps are dense, so d@1 is
  structurally 0 — the informative within-cycle horizons start at H=2.

Evaluation: distortion(H) = ||w_sparse - w_dense||/||w_dense - w0|| against
the dense replay arm; NDCG@k / recall@k / missed_label_mass /
missed_score_mass against the dense realized per-coordinate squared
gradient-driven updates (label).

Numerical semantics (shared with the legacy tool, pinned by
tests/test_replay_cycle.py):
- the logged gradient is post-unscale/post-clip; sparse replays reconstruct
  pre-clip via g / coef_dense with coef_dense = min(1, clip/(norm+1e-6)),
  then re-clip the zero-filled/folded q per step by the same formula on its
  own norm;
- the commit consuming g at global step t is commit number t+1 (bias
  correction exponent at the branch point is t0);
- top-k selections break ties by index (catalog order, then row-major),
  the threshold coming from the exact 32769-bin bit-pattern histogram select
  (bit-identical to torch.kthvalue over the flat score vector).

Hyperparameters: legacy traces carry GLOBAL adam_beta1/beta2/eps/weight_decay
plus a per-group lr list (this tool replays every param with the global
values and the per-step MEAN lr — the legacy behavior). New traces may
additionally carry per-group hyperparameters (meta["param_groups"] +
catalog[name]["param_group"]); when present they take precedence per param
(exact lr per group included). The fallback keeps old-trace replays
bit-compatible with the legacy tool.

Usage (SGA repo root on PYTHONPATH):

    python -m sga.replay --trace-dir <dir> --branch-point 40 \
        --replay-mode cycle --cycle-len 20 --ratios 10,20 --horizons 1,10 \
        --score-names abs_m,update --dense-cache <cache-dir> --out report.json

    # replay-machinery validation against real state dumps:
    python -m sga.replay --trace-dir <dir> --self-check 40 60

    # offline acceleration: bf16 screening copy + dense-artifact cache
    python -m sga.trace_convert --src <fp32-trace> --dst <bf16-trace>
    python -m sga.replay --trace-dir <bf16-trace> ... --trace-dtype bf16 \
        --dense-cache <cache-dir>
"""

import argparse
import glob
import hashlib
import json
import math
import os
import re
import time

import torch


from sga.plugins.scores import get_score_spec, score_names as all_score_names

# Bump on ANY change to replay arithmetic or semantics (score formulas,
# clip/norm handling, dtype rules, ...). Part of the dense-cache key: a stale
# cache from an older semantics must fail loudly, never silently serve.
REPLAY_SEMANTIC_VERSION = 2

_G_STEP_RE = re.compile(r"step_(\d+)_g\.pt$")
_STAT_STEP_RE = re.compile(r"step_(\d+)_state\.pt$")


# ----------------------------------------------------------------------------
# loading / trace discovery
# ----------------------------------------------------------------------------

def _load(path, mmap=True):
    """Load a trace file. ``weights_only=False`` + mmap to avoid copying the
    5-20 GB tensors into heap (storages stay file-backed)."""
    return torch.load(path, map_location="cpu", weights_only=False, mmap=mmap)


def _load_g(g_files, gstep, upcast_bf16=False):
    """Load one step's g dump. With ``upcast_bf16`` (--trace-dtype bf16
    screening copies) bf16 payloads are upcast to fp32, so every replay path
    consumes exactly fp32 inputs. A bf16 payload met with the upcast off is a
    hard error: the sparse paths would divide by coef_dense in bf16 and
    silently lose precision while the report still records the requested
    fp32 declaration."""
    gdict = _load(g_files[gstep])
    gp = gdict["payload"]
    bf16_keys = [k for k, t in gp.items()
                 if torch.is_tensor(t) and t.dtype == torch.bfloat16]
    if bf16_keys and not upcast_bf16:
        raise ValueError(
            f"{g_files[gstep]}: bf16 payload(s) {bf16_keys} but "
            "--trace-dtype is fp32; rerun with --trace-dtype bf16 "
            "(payloads are upcast to fp32 at load)")
    for k in bf16_keys:
        gp[k] = gp[k].float()
    return gdict


def discover_trace(trace_dir):
    """Return (catalog dict, ordered names, {name: offsets}, N,
    {step: state_path}, {step: g_path})."""
    if not os.path.isdir(trace_dir):
        raise ValueError(f"--trace-dir does not exist: {trace_dir}")
    cat_path = os.path.join(trace_dir, "catalog.pt")
    if not os.path.isfile(cat_path):
        raise ValueError(f"no catalog.pt in {trace_dir}")
    cat = _load(cat_path, mmap=False)
    catalog = cat["catalog"]
    names = list(catalog.keys())
    if not names:
        raise ValueError(f"catalog.pt has no params in {trace_dir}")
    offsets = {}
    N = 0
    for n in names:
        offsets[n] = N
        nelt = 1
        for s in catalog[n]["shape"]:
            nelt *= int(s)
        catalog[n]["numel"] = nelt
        N += nelt
    state_files = {}
    g_files = {}
    for f in sorted(glob.glob(os.path.join(trace_dir, "*.pt"))):
        base = os.path.basename(f)
        m = _G_STEP_RE.match(base)
        if m:
            g_files[int(m.group(1))] = f
            continue
        m = _STAT_STEP_RE.match(base)
        if m:
            state_files[int(m.group(1))] = f
    if not state_files:
        raise ValueError(f"no step_*_state.pt files in {trace_dir}")
    return catalog, names, offsets, N, state_files, g_files


def window_steps_for(g_files, t0, window):
    """Contiguous run of logged global steps starting at t0 (clipped to
    ``window`` length, stopping at the first missing g file)."""
    steps = []
    for s in range(t0, t0 + window):
        if s in g_files:
            steps.append(s)
        else:
            break
    return steps


def parse_branch_spec(spec, state_files):
    """--branch-point: comma-list of ints, or the literal ``all`` (every
    state file found). Returns a sorted list of ints."""
    s = spec.strip()
    if s.lower() == "all":
        return sorted(state_files.keys())
    pts = []
    for tok in s.replace(" ", "").split(","):
        if not tok:
            continue
        pts.append(int(tok))
    pts = sorted(set(pts))
    missing = [p for p in pts if p not in state_files]
    if missing:
        raise ValueError(
            f"branch point(s) {missing} have no step_NNNNNN_state.pt in the trace dir"
        )
    return pts


def scalar_list(v):
    return [float(x) for x in v]


def _lr_for_step(gmeta, group=None):
    """Per-step learning rate. New traces record one lr per optimizer param
    group; with a known group index the exact value is used, otherwise the
    legacy mean-over-groups fallback applies."""
    lrs = scalar_list(gmeta.get("lr", [0.0]))
    if not lrs:
        return 0.0
    if group is not None and group < len(lrs):
        return lrs[group]
    return sum(lrs) / len(lrs)


def _lr_max_abs_diff(gmeta):
    lrs = scalar_list(gmeta.get("lr", []))
    if not lrs:
        return 0.0
    mean = sum(lrs) / len(lrs)
    return max(abs(l - mean) for l in lrs)


def _param_hp(meta, catalog_entry):
    """Resolve one param's (beta1, beta2, eps, weight_decay, group_index).

    New traces carry meta["param_groups"] (list of per-group hyperparameter
    dicts) and catalog[name]["param_group"]; legacy traces carry neither and
    fall back to the global meta scalars (legacy replay behavior, which keeps
    old-trace results bit-comparable with the legacy tool).
    """
    groups = meta.get("param_groups") if isinstance(meta, dict) else None
    gi = catalog_entry.get("param_group") if catalog_entry else None
    if groups is not None and gi is not None:
        gi = int(gi)
        if not 0 <= gi < len(groups):
            # A raw IndexError here would be cryptic; the catalog/meta pair
            # disagreeing is a trace-integrity problem and must say so.
            raise ValueError(
                f"trace catalog param_group index {gi} outside the step "
                f"meta param_groups (0..{len(groups) - 1}); catalog and meta "
                "come from different runs"
            )
        g = groups[gi]
        return (float(g["adam_beta1"]), float(g["adam_beta2"]),
                float(g["adam_eps"]), float(g["weight_decay"]), gi)
    return (float(meta["adam_beta1"]), float(meta["adam_beta2"]),
            float(meta["adam_eps"]), float(meta["weight_decay"]), None)


# ----------------------------------------------------------------------------
# score computation (fp64 arithmetic, fp32 storage)
# ----------------------------------------------------------------------------

def _chunk_elems(chunk_mb, max_bytes_per_elem):
    "Element count per chunk for the fp64 working set (8 bytes/elem)."
    return max(1, int(chunk_mb * (2 ** 20) // max_bytes_per_elem))


def compute_scores_flat(w_parts, m_parts, v_parts, g_parts, offsets, N, meta,
                        chunk_mb, device="cpu", score_names=None,
                        update_includes_wd=False, hp_parts=None):
    """Per-coordinate scores over the whole model into flat fp32 tensors.

    All chunk arithmetic is carried in torch.float64. ``g`` is the logged
    (post-unscale/post-clip) gradient consumed by the commit at step t0.

    ``update_includes_wd=True`` (the online-exact full-sync selection
    score) switches ``update`` to |wd*w + U(g)| — the materialized
    grad-correct update of the upcoming commit; the default reproduces the
    original offline scores.

    ``hp_parts`` (optional) maps param name -> (beta1, beta2, eps, wd) for
    per-group hyperparameters (new traces); absent, the global meta scalars
    apply to every param (legacy behavior).
    """
    score_names = score_names or list(all_score_names())
    specs = {s: get_score_spec(s) for s in score_names}
    gb1, gb2 = float(meta["adam_beta1"]), float(meta["adam_beta2"])
    geps, gwd = float(meta["adam_eps"]), float(meta["weight_decay"])
    t0 = int(meta["step"])
    chunk_n = _chunk_elems(chunk_mb, 8)
    dev = "cpu" if device == "cpu" else device

    out = {s: torch.empty(N, dtype=torch.float32) for s in score_names}
    for name, w_ in w_parts:
        o = offsets[name]
        if hp_parts is not None and name in hp_parts:
            b1, b2, eps, wd = hp_parts[name]
        else:
            b1, b2, eps, wd = gb1, gb2, geps, gwd
        # ``state_absent`` implies t0 == 0 and m == v == 0; the formulas take
        # the bias corrections of the UPCOMING commit (exponent t0 + 1).
        bc1p = 1.0 - b1 ** (t0 + 1)
        bc2p = 1.0 - b2 ** (t0 + 1)
        w = w_.reshape(-1)
        n = w.numel()
        m = m_parts[name].reshape(-1)
        v = v_parts[name].reshape(-1)
        g = g_parts[name].reshape(-1)
        for a in range(0, n, chunk_n):
            b = min(n, a + chunk_n)
            w64 = w[a:b].to(dtype=torch.float64, device=dev)
            m64 = m[a:b].to(dtype=torch.float64, device=dev)
            v64 = v[a:b].to(dtype=torch.float64, device=dev)
            g64 = g[a:b].to(dtype=torch.float64, device=dev)
            for s in score_names:
                # the materialized form may be signed (update) — rank by magnitude
                sc = specs[s].formula(w64, m64, v64, g64, b1, b2, eps, wd,
                                      bc1p, bc2p,
                                      update_includes_wd=update_includes_wd)
                out[s][o + a: o + b] = sc.abs()
    return out


def _score_stats(flat):
    total = flat.numel()
    ssum = float(flat.sum())
    zmask = flat == 0
    frac0 = float(zmask.sum()) / total if total else 0.0
    return {
        "numel": total,
        "sum": ssum,
        "mean": ssum / total if total else 0.0,
        "min": float(flat.min()) if total else 0.0,
        "max": float(flat.max()) if total else 0.0,
        "fraction_zero": frac0,
    }


# ----------------------------------------------------------------------------
# exact global top-k support (ties by index) + ranking metrics
# ----------------------------------------------------------------------------

def _abs_bit_top16(t):
    """Top 16 bits of the |x| fp32 bit pattern (int32, monotone in |x|);
    non-negative fp32 values sort identically to their int32 bit patterns."""
    return ((t.view(torch.int32) & 0x7FFFFFFF) >> 16).view(-1)


def hist_threshold(t, k):
    """Exact k-th-largest value over the flat score vector via a 32769-bin
    bit-pattern histogram + candidate-bin kthvalue. Bit-exact same threshold
    as torch.kthvalue over all elements (the trick used by the online
    Phase-3 selection).
    ``t`` must be non-negative (all our scores are absolute values)."""
    top16 = _abs_bit_top16(t)
    hist = torch.bincount(top16, minlength=32769)
    cum = torch.cumsum(hist, 0)
    total = int(cum[-1].item())
    r = total - k + 1
    b = int(torch.searchsorted(cum, r).item())
    n_above = total - int(cum[b].item())
    cand = t[top16 == b].abs()
    kk = cand.numel() - (k - n_above) + 1
    if cand.numel() == 0 or not (1 <= kk <= cand.numel()):
        raise RuntimeError(
            f"hist select math out of range: cand={cand.numel()} kk={kk} r={r}")
    return float(torch.kthvalue(cand, kk).values.item())


def select_topk_flat(t, k):
    """Exact top-k on the globally concatenated fp32 score vector.

    The k-th-largest threshold comes from the exact bit-pattern histogram
    select (bit-identical to torch.kthvalue); strictly-above coords are
    selected and the boundary tie layer is filled in global index order
    (catalog param order, then row-major within-param index), so the result
    is fully deterministic. Requires non-negative scores (the exact bit-pattern
    histogram select is magnitude-based). Returns (bool flat mask, threshold)."""
    N = t.numel()
    if k > N:
        raise ValueError(f"k={k} > N={N}")
    if float(t.min()) < 0.0:
        raise ValueError("select_topk_flat requires non-negative scores")
    if not bool(torch.isfinite(t).all()):
        # NaN/Inf sit in the histogram's top bit-pattern bins: they would be
        # selected FIRST, silently displacing real coordinates from the topk.
        raise ValueError(
            "select_topk_flat requires finite scores (NaN/Inf would land in "
            "the top bins and silently pollute the topk)"
        )
    thresh = hist_threshold(t, k)
    gt = t > thresh
    n_gt = int(gt.sum().item())
    need = k - n_gt
    if need > 0:
        eq = t == thresh
        # first ``need`` tied coords, by index: only the tie positions whose
        # running tie-count (including themselves) does not exceed ``need``.
        head = eq & (eq.cumsum(0) <= need)
        gt = gt | head
    n_sel = int(gt.sum().item())
    if n_sel != k:
        raise RuntimeError(f"internal tie-break error: selected {n_sel} != {k}")
    return gt, thresh


def ranking_metrics(sc_flat, label_flat, sel, label_sel, k, total_score,
                    total_label, idcg):
    """NDCG@k (linear gain), recall@k vs the label top-k set, and both missed
    masses. ``sc_flat``/``label_flat`` are the concatenated fp32 score and
    label vectors (catalog order); ``sel`` the score top-k mask,
    ``label_sel`` the label top-k mask for the same k and ``idcg`` its
    precomputed ideal DCG (identical for every score of a ratio)."""
    sc_sel = sc_flat[sel]
    lb_sel = label_flat[sel]
    pos = torch.arange(1, k + 1, dtype=torch.float64)
    wts = 1.0 / torch.log2(pos + 1.0)
    order = torch.argsort(sc_sel, descending=True, stable=True)
    dcg = float((lb_sel[order].to(torch.float64) * wts).sum())
    if idcg > 0.0:
        ndcg = dcg / idcg
    else:
        ndcg = 1.0 if dcg == 0.0 else 0.0

    recall = int(sel.logical_and(label_sel).sum().item()) / float(k)

    sel_score_sum = float(sc_sel.sum())
    sel_label_sum = float(lb_sel.sum())
    if total_label > 0:
        missed_label = 1.0 - sel_label_sum / total_label
    else:
        missed_label = 0.0 if sel_label_sum == 0 else 1.0
    if total_score > 0:
        missed_score = 1.0 - sel_score_sum / total_score
    else:
        missed_score = 0.0 if sel_score_sum == 0 else 1.0

    return {
        "ndcg_at_k": ndcg,
        "recall_at_k": recall,
        "missed_label_mass": missed_label,
        "missed_score_mass": missed_score,
    }


def idcg_for(label_flat, label_sel, k):
    """Ideal DCG of the top-k label set (largest possible value of the DCG
    over any k-subset, achieved by picking the k largest labels)."""
    lbk = label_flat[label_sel]
    rel2 = torch.sort(lbk, descending=True, stable=True).values.to(torch.float64)
    pos = torch.arange(1, k + 1, dtype=torch.float64)
    wts = 1.0 / torch.log2(pos + 1.0)
    return float((rel2 * wts).sum())


# ----------------------------------------------------------------------------
# AdamW replay (per-param streaming, chunked)
# ----------------------------------------------------------------------------

def coef_dense_for(meta):
    """Applied clip coefficient mirroring the trainer's fp32 total-norm clip:
    clip_coeff = max_norm / (total_norm + 1.0e-6), clamped to <= 1.0
    (grads left untouched when the coefficient is >= 1).
    A NaN/unavailable grad_norm or a non-positive clip_grad means nobody
    clipped -> coefficient 1.0."""
    clip = float(meta.get("clip_grad", 0.0))
    norm = float(meta.get("grad_norm", float("nan")))
    flag = None
    if clip <= 0.0:
        flag = "clipping_disabled"
        return 1.0, flag
    if not math.isfinite(norm) or norm <= 0.0:
        flag = "grad_norm_unavailable"
        return 1.0, flag
    coef = clip / (norm + 1.0e-6)
    coef = min(coef, 1.0)
    return coef, flag


def adam_update_chunk(q, m, v, w, lr, wd, b1, b2, bc1c, bc2c, eps, device):
    """One AdamW commit on a chunk, returning the new (m, v, w) cpu tensors
    plus the squared LR-scaled gradient-driven update (label term)."""
    qd = q.to(device)
    md = m.to(device)
    vd = v.to(device)
    wd_t = w.to(device)
    mn = b1 * md + (1.0 - b1) * qd
    vn = b2 * vd + (1.0 - b2) * qd * qd
    mh = mn / bc1c
    vh = vn / bc2c
    upd = mh / (torch.sqrt(vh) + eps)
    wn = wd_t * (1.0 - lr * wd) - lr * upd
    upd2 = (lr * upd) * (lr * upd)
    if device == "cpu":
        return mn, vn, wn, upd2
    return mn.cpu(), vn.cpu(), wn.cpu(), upd2.cpu()


def _adam_step_inplace(q, m, v, w, lab, lr, wd, b1, b2, bc1c, bc2c, eps):
    """One AdamW commit on device-resident chunks, in place.
    q/m/v/w: same-shaped device tensors; lab: optional device tensor
    accumulating (lr * gradient-driven update)^2."""
    m.mul_(b1).add_(q, alpha=1.0 - b1)
    v.mul_(b2).addcmul_(q, q, value=1.0 - b2)
    mh = m / bc1c
    vh = torch.sqrt(v / bc2c)
    upd = mh / (vh + eps)
    w.mul_(1.0 - lr * wd).add_(upd, alpha=-lr)
    if lab is not None:
        lab.addcmul_(upd, upd, value=lr * lr)


def _replay_meta(params, base_meta):
    """(names, offsets, N) for the replay loops; per-param hyperparameters
    live on the prm dicts (build_branch_state)."""
    return (
        [p["name"] for p in params],
        {p["name"]: p["off"] for p in params},
        sum(p["n"] for p in params),
    )


def dense_replay(params, g_files, window_steps, base_meta, t0, eff_horizons,
                 replay_dtype, chunk_mb, device, upcast_bf16=False):
    """Dense trajectory w_d / m_d / v_d from the logged (post-clip) g, the
    per-coordinate squared gradient-driven update label, the per-horizon
    denominator norms ||w_d(H)-w0||, pre-clip reconstruction diagnostics, and
    per-horizon snapshot copies of w_d (used as the sparse-replay reference
    so sparse replays only carry the sparse m/v/w states).

    ``params`` = list of {name, w0, m0, v0, w, m, v, label, off, n, b1, b2,
    eps, wd, group} where w/m/v live in replay_dtype; w0 is the immutable
    branch-start reference. Mutates the ``w/m/v/label`` entries. Returns
    (label_total, den_at, recon_steps, snap_d).
    """
    (names, offs, N) = _replay_meta(params, base_meta)
    chunk_n = _chunk_elems(chunk_mb, torch.finfo(replay_dtype).bits // 8)
    den_at = {}
    snap_d = {}
    recon_steps = []
    label_total = 0.0

    for si, gstep in enumerate(window_steps):
        gdict = _load_g(g_files, gstep, upcast_bf16)
        gmeta = gdict["meta"]
        gp = gdict["payload"]
        coef_dense, flag = coef_dense_for(gmeta)
        # commit count at global step t is t+1
        count = t0 + si + 1
        recon2 = 0.0
        for prm in params:
            name, o, n = prm["name"], prm["off"], prm["n"]
            lr = _lr_for_step(gmeta, prm["group"])
            b1, b2, eps, wd = prm["b1"], prm["b2"], prm["eps"], prm["wd"]
            bc1c = 1.0 - b1 ** count
            bc2c = 1.0 - b2 ** count
            gt = gp.get(name)
            if gt is None:
                g = torch.zeros(n, dtype=torch.float32)
            else:
                g = gt.reshape(-1)
            m, v, w, lab = prm["m"], prm["v"], prm["w"], prm["label"]
            for a in range(0, n, chunk_n):
                b = min(n, a + chunk_n)
                q = g[a:b].to(replay_dtype)
                mn, vn, wn, upd2 = adam_update_chunk(
                    q, m[a:b], v[a:b], w[a:b],
                    lr, wd, b1, b2, bc1c, bc2c, eps, device,
                )
                m[a:b] = mn
                v[a:b] = vn
                w[a:b] = wn
                lab[a:b] += upd2
                recon2 += float(((q.double() / coef_dense) ** 2.0).sum())
        recon_norm = math.sqrt(recon2)
        recon_steps.append({
            "step": gstep, "lr": _lr_for_step(gmeta),
            "lr_max_abs_diff": _lr_max_abs_diff(gmeta),
            "grad_norm": float(gmeta.get("grad_norm", float("nan"))),
            "clip_grad": float(gmeta.get("clip_grad", 0.0)),
            "coef_dense": coef_dense,
            "reconstruct_flag": flag,
            "recon_grad_norm": recon_norm,
            "recon_rel_err": (
                (recon_norm - float(gmeta.get("grad_norm", float("nan"))))
                / float(gmeta.get("grad_norm", 1.0))
                if math.isfinite(float(gmeta.get("grad_norm", float("nan"))))
                and float(gmeta.get("grad_norm", 0.0)) > 0.0 else None
            ),
        })
        if (si + 1) in eff_horizons:
            d2 = 0.0
            snap = {}
            for prm in params:
                w = prm["w"]
                w0 = prm["w0"]
                n = prm["n"]
                # chunked to keep the diff temp small even for huge params
                for a in range(0, n, chunk_n):
                    b = min(n, a + chunk_n)
                    d = (w[a:b] - w0[a:b]).to(torch.float32)
                    d2 += float((d.to(torch.float64) ** 2).sum())
                snap[prm["name"]] = w.clone()
            den_at[si + 1] = math.sqrt(d2)
            snap_d[si + 1] = snap

    label_total = 0.0
    for prm in params:
        label_total += float(prm["label"].to(torch.float64).sum())
    return label_total, den_at, recon_steps, snap_d


def dense_replay_dev(params, g_files, window_steps, base_meta, t0, eff_horizons,
                     replay_dtype, chunk_mb, device, upcast_bf16=False):
    """Device-resident dense replay; same outputs as dense_replay but the
    dense m/v/w/label never leave ``device`` (horizon snapshots are cloned to
    cpu). Exists to avoid the 40-80 GB/step H2D/D2H round trips of the
    cpu-state path at 1B+ scale."""
    (names, offs, N) = _replay_meta(params, base_meta)
    chunk_n = _chunk_elems(chunk_mb, torch.finfo(replay_dtype).bits // 8)
    dev = device
    for prm in params:
        prm["w"] = prm["w0"].to(dev)
        prm["m"] = prm["m0"].to(dev)
        prm["v"] = prm["v0"].to(dev)
        prm["label"] = torch.zeros(prm["n"], dtype=replay_dtype, device=dev)
    den_at, snap_d, recon_steps = {}, {}, []

    for si, gstep in enumerate(window_steps):
        gdict = _load_g(g_files, gstep, upcast_bf16)
        gmeta = gdict["meta"]
        gp = gdict["payload"]
        coef_dense, flag = coef_dense_for(gmeta)
        count = t0 + si + 1
        recon2 = 0.0
        for prm in params:
            name, n = prm["name"], prm["n"]
            lr = _lr_for_step(gmeta, prm["group"])
            b1, b2, eps, wd = prm["b1"], prm["b2"], prm["eps"], prm["wd"]
            bc1c = 1.0 - b1 ** count
            bc2c = 1.0 - b2 ** count
            gt = gp.get(name)
            g = torch.zeros(n, dtype=torch.float32) if gt is None else gt.reshape(-1)
            m, v, w, lab = prm["m"], prm["v"], prm["w"], prm["label"]
            for a in range(0, n, chunk_n):
                b = min(n, a + chunk_n)
                q = g[a:b].to(dev)
                if replay_dtype != torch.float32:
                    q = q.to(replay_dtype)
                _adam_step_inplace(q, m[a:b], v[a:b], w[a:b], lab[a:b],
                                   lr, wd, b1, b2, bc1c, bc2c, eps)
                recon2 += float(((q.double() / coef_dense) ** 2.0).sum())
        recon_norm = math.sqrt(recon2)
        recon_steps.append({
            "step": gstep, "lr": _lr_for_step(gmeta),
            "lr_max_abs_diff": _lr_max_abs_diff(gmeta),
            "grad_norm": float(gmeta.get("grad_norm", float("nan"))),
            "clip_grad": float(gmeta.get("clip_grad", 0.0)),
            "coef_dense": coef_dense, "reconstruct_flag": flag,
            "recon_grad_norm": recon_norm,
            "recon_rel_err": (
                (recon_norm - float(gmeta.get("grad_norm", float("nan"))))
                / float(gmeta.get("grad_norm", 1.0))
                if math.isfinite(float(gmeta.get("grad_norm", float("nan"))))
                and float(gmeta.get("grad_norm", 0.0)) > 0.0 else None
            ),
        })
        if (si + 1) in eff_horizons:
            d2 = 0.0
            snap = {}
            for prm in params:
                w, w0, n = prm["w"], prm["w0"], prm["n"]
                for a in range(0, n, chunk_n):
                    b = min(n, a + chunk_n)
                    d = (w[a:b] - w0[a:b].to(dev)).double()
                    d2 += float((d * d).sum())
                snap[prm["name"]] = w.cpu()
            den_at[si + 1] = math.sqrt(d2)
            snap_d[si + 1] = snap
    label_total = 0.0
    for prm in params:
        prm["label"] = prm["label"].cpu()
        label_total += float(prm["label"].to(torch.float64).sum())
    # free the dense running states; keep w0/m0/v0 (cpu) for sparse replays
    for prm in params:
        for key in ("w", "m", "v"):
            prm.pop(key, None)
    if dev != "cpu":
        torch.cuda.empty_cache()
    return label_total, den_at, recon_steps, snap_d


def sparse_replay(params, g_files, window_steps, base_meta, t0, eff_horizons,
                  sel, snap_d, replay_dtype, chunk_mb, device,
                  upcast_bf16=False, norm_fp32=False):
    """Sparse replay with a fixed support ``sel`` (flat bool, catalog order).

    Per window step the pre-clip gradient is reconstructed
    (pre_clip = g_logged / coef_dense), zero-filled outside the support, and
    re-clipped with the same coefficient formula on its own global norm;
    then one AdamW commit is applied with q = pre_clip * coef_sparse.

    Numerator norms num(H) = ||w_s(H) - w_d(H)|| are measured against the
    frozen dense snapshots ``snap_d`` taken by ``dense_replay`` at each
    effective horizon, so the sparse replay only needs the sparse states.

    Returns {h: num(H)}.
    """
    (names, offs, N) = _replay_meta(params, base_meta)
    chunk_n = _chunk_elems(chunk_mb, torch.finfo(replay_dtype).bits // 8)
    num_at = {}
    for prm in params:
        prm["ms"] = prm["m0"].clone()
        prm["vs"] = prm["v0"].clone()
        prm["ws"] = prm["w0"].clone()
    for si, gstep in enumerate(window_steps):
        gdict = _load_g(g_files, gstep, upcast_bf16)
        gp = gdict["payload"]
        coef_dense, _flag = coef_dense_for(gdict["meta"])
        gmeta = gdict["meta"]
        # -- pass 1: global norm of the zero-filled pre-clip q (fp32 fast) ----
        qn2 = 0.0
        for prm in params:
            name, o, n = prm["name"], prm["off"], prm["n"]
            gt = gp.get(name)
            if gt is None:
                continue
            g = gt.reshape(-1)
            for a in range(0, n, chunk_n):
                b = min(n, a + chunk_n)
                q_pre = (g[a:b] / coef_dense) * sel[o + a: o + b].to(torch.bool)
                if norm_fp32:
                    qn2 += float((q_pre * q_pre).sum())
                else:
                    qn2 += float((q_pre.double() ** 2).sum())
        qnorm = math.sqrt(qn2)
        clip_grad = float(gmeta.get("clip_grad", 0.0))
        coef_sparse = 1.0
        if clip_grad > 0.0:
            coef_sparse = min(1.0, clip_grad / (qnorm + 1.0e-6))
        # -- pass 2: apply the sparse commit ----------------------------------
        count = t0 + si + 1
        for prm in params:
            name, o, n = prm["name"], prm["off"], prm["n"]
            lr = _lr_for_step(gmeta, prm["group"])
            b1, b2, eps, wd = prm["b1"], prm["b2"], prm["eps"], prm["wd"]
            bc1c = 1.0 - b1 ** count
            bc2c = 1.0 - b2 ** count
            gt = gp.get(name)
            # params without a logged grad contribute zero q (their moments
            # still decay and weight decay still applies)
            if gt is None:
                g = torch.zeros(n, dtype=torch.float32)
            else:
                g = gt.reshape(-1)
            ms, vs, ws = prm["ms"], prm["vs"], prm["ws"]
            for a in range(0, n, chunk_n):
                b = min(n, a + chunk_n)
                gd = (g[a:b] / coef_dense).to(replay_dtype)
                q = gd * sel[o + a: o + b].to(replay_dtype) * coef_sparse
                m2, v2, w2, _u = adam_update_chunk(
                    q, ms[a:b], vs[a:b], ws[a:b],
                    lr, wd, b1, b2, bc1c, bc2c, eps, device,
                )
                ms[a:b] = m2
                vs[a:b] = v2
                ws[a:b] = w2
        if (si + 1) in eff_horizons:
            num2 = 0.0
            snap = snap_d[si + 1]
            for prm in params:
                ws = prm["ws"]
                wd_ = snap[prm["name"]]
                n = prm["n"]
                for a in range(0, n, chunk_n):
                    b = min(n, a + chunk_n)
                    d = (ws[a:b] - wd_[a:b]).to(torch.float32)
                    num2 += float((d.to(torch.float64) ** 2).sum())
            num_at[si + 1] = math.sqrt(num2)
    return num_at


def sparse_replay_dev(params, g_files, window_steps, base_meta, t0, eff_horizons,
                      sel, snap_d, replay_dtype, chunk_mb, device,
                      upcast_bf16=False, norm_fp32=False):
    """Device-resident sparse replay; semantics identical to sparse_replay."""
    (names, offs, N) = _replay_meta(params, base_meta)
    chunk_n = _chunk_elems(chunk_mb, torch.finfo(replay_dtype).bits // 8)
    dev = device
    for prm in params:
        prm["ms"] = prm["m0"].to(dev)
        prm["vs"] = prm["v0"].to(dev)
        prm["ws"] = prm["w0"].to(dev)
    num_at = {}
    for si, gstep in enumerate(window_steps):
        gdict = _load_g(g_files, gstep, upcast_bf16)
        gp = gdict["payload"]
        gmeta = gdict["meta"]
        coef_dense, _flag = coef_dense_for(gmeta)
        # pass 1: global norm of zero-filled pre-clip q
        qn2 = 0.0
        for prm in params:
            name, o, n = prm["name"], prm["off"], prm["n"]
            gt = gp.get(name)
            if gt is None:
                continue
            g = gt.reshape(-1)
            for a in range(0, n, chunk_n):
                b = min(n, a + chunk_n)
                q_pre = (g[a:b].to(dev) / coef_dense) * sel[o + a: o + b].to(dev)
                if norm_fp32:
                    qn2 += float((q_pre * q_pre).sum())
                else:
                    qn2 += float((q_pre.double() ** 2.0).sum())
        qnorm = math.sqrt(qn2)
        clip_grad = float(gmeta.get("clip_grad", 0.0))
        coef_sparse = 1.0
        if clip_grad > 0.0:
            coef_sparse = min(1.0, clip_grad / (qnorm + 1.0e-6))
        # pass 2: apply the sparse commit on all coords (q masked)
        count = t0 + si + 1
        for prm in params:
            name, o, n = prm["name"], prm["off"], prm["n"]
            lr = _lr_for_step(gmeta, prm["group"])
            b1, b2, eps, wd = prm["b1"], prm["b2"], prm["eps"], prm["wd"]
            bc1c = 1.0 - b1 ** count
            bc2c = 1.0 - b2 ** count
            gt = gp.get(name)
            g = torch.zeros(n, dtype=torch.float32) if gt is None else gt.reshape(-1)
            ms, vs, ws = prm["ms"], prm["vs"], prm["ws"]
            for a in range(0, n, chunk_n):
                b = min(n, a + chunk_n)
                q = (g[a:b].to(dev) / coef_dense) * sel[o + a: o + b].to(dev) * coef_sparse
                if replay_dtype != torch.float32:
                    q = q.to(replay_dtype)
                _adam_step_inplace(q, ms[a:b], vs[a:b], ws[a:b], None,
                                   lr, wd, b1, b2, bc1c, bc2c, eps)
        if (si + 1) in eff_horizons:
            num2 = 0.0
            snap = snap_d[si + 1]
            for prm in params:
                ws, wd_, n = prm["ws"], snap[prm["name"]], prm["n"]
                for a in range(0, n, chunk_n):
                    b = min(n, a + chunk_n)
                    d = (ws[a:b] - wd_[a:b].to(dev)).double()
                    num2 += float((d * d).sum())
            num_at[si + 1] = math.sqrt(num2)
    for prm in params:
        for key in ("ms", "vs", "ws"):
            prm.pop(key, None)
    if dev != "cpu":
        torch.cuda.empty_cache()
    return num_at



# ----------------------------------------------------------------------------
# dense replay artifact cache (--dense-cache)
# ----------------------------------------------------------------------------

def _dense_cache_ident(trace_dir, t0, window_steps, eff_horizons,
                       replay_dtype, trace_dtype, dense_engine):
    """Identity of one branch's dense replay artifacts. The trace path stands
    for the trace content (dumps are immutable); window/horizons/dtype change
    the saved artifacts; dense_engine fences the two dense engines apart
    (dense_replay quantizes horizon diffs to fp32 before squaring,
    dense_replay_dev does not — a shared ident would silently mix them);
    REPLAY_SEMANTIC_VERSION fences caches across replay semantic edits."""
    return "|".join([
        f"semver={REPLAY_SEMANTIC_VERSION}",
        f"trace={trace_dir}",
        f"t0={t0}",
        f"W={len(window_steps)}",
        f"horizons={','.join(str(h) for h in eff_horizons)}",
        f"dtype={replay_dtype}",
        f"trace_dtype={trace_dtype}",
        f"dense_engine={dense_engine}",
    ])


def _dense_cache_path(cache_dir, ident):
    h = hashlib.sha1(ident.encode()).hexdigest()[:16]
    return os.path.join(cache_dir, f"dense_{h}.pt")


def save_dense_cache(cache_dir, ident, label_total, den_at, recon_steps,
                     snap_d, label_flat):
    """Persist one branch's dense artifacts atomically (tmp + rename), so a
    killed writer never leaves a half file that a later run mistakes for a
    usable cache. Returns the cache file path."""
    blob = {
        "ident": ident,
        "label_total": label_total,
        "den_at": den_at,
        "recon_steps": recon_steps,
        "snap_d": snap_d,
        "label_flat": label_flat,
    }
    os.makedirs(cache_dir, exist_ok=True)
    path = _dense_cache_path(cache_dir, ident)
    tmp = f"{path}.tmp.{os.getpid()}"
    torch.save(blob, tmp)
    os.replace(tmp, path)
    return path


def load_dense_cache(cache_dir, ident):
    """Load the dense artifacts for ``ident``; None on miss. A present file
    whose ident disagrees (hand-copied/hand-edited cache) fails loudly —
    silently replaying against someone else's dense trajectory would corrupt
    every distortion number this run reports."""
    path = _dense_cache_path(cache_dir, ident)
    if not os.path.isfile(path):
        return None
    try:
        blob = _load(path)
    except Exception as e:
        # a truncated/corrupt blob (host/NFS crash mid-write) would otherwise
        # surface as an opaque mmap RuntimeError or AttributeError
        raise ValueError(
            f"dense cache {path} is unreadable ({e!r}); "
            "delete it and rerun") from e
    if blob.get("ident") != ident:
        raise ValueError(
            f"dense cache {path} holds ident\n  {blob.get('ident')!r}\n"
            f"but this run needs\n  {ident!r}\n"
            "clear the cache dir and rerun")
    return {"path": path, **blob}


# ----------------------------------------------------------------------------
# branch processing
# ----------------------------------------------------------------------------

def build_branch_state(state_file, names, replay_dtype, catalog=None):
    """Load w/m/v/g at the branch point (mmap-file-backed views -> cloned into
    replay_dtype copies for w0/m0/v0; g left fp32). Per-param hyperparameters
    resolve from ``catalog`` (param_group) + the state's meta (param_groups)
    when present, else from the global meta scalars (legacy behavior)."""
    st = _load(state_file)
    meta = st["meta"]
    w_parts, m_parts, v_parts, g_parts = [], {}, {}, {}
    params = []
    off = 0
    for name in names:
        pw = st["payload"]["w"].get(name)
        if pw is None:
            raise ValueError(f"state file {state_file} missing w for {name}")
        # per-param tensors are stored flat (row-major) so chunk offsets
        # match the global flat index space used by the support masks
        w0 = pw.to(replay_dtype).reshape(-1).clone()
        m0 = st["payload"]["m"].get(name)
        if m0 is None:
            m0 = torch.zeros_like(pw)
        m0 = m0.to(replay_dtype).reshape(-1).clone()
        v0 = st["payload"]["v"].get(name)
        if v0 is None:
            v0 = torch.zeros_like(pw)
        v0 = v0.to(replay_dtype).reshape(-1).clone()
        g = st["payload"].get("g", {}).get(name)
        if g is None:
            g = torch.zeros_like(pw)
        g = g.float().reshape(-1)
        n = pw.numel()
        b1, b2, eps, wd, group = _param_hp(
            meta, catalog.get(name) if catalog else None)
        params.append({
            "name": name,
            "off": off,
            "n": n,
            "w0": w0,
            "m0": m0,
            "v0": v0,
            "b1": b1,
            "b2": b2,
            "eps": eps,
            "wd": wd,
            "group": group,
        })
        w_parts.append((name, w0))
        m_parts[name] = m0
        v_parts[name] = v0
        g_parts[name] = g
        off += n
    return meta, params, w_parts, m_parts, v_parts, g_parts


def process_branch(t0, ctx, args):
    """Analyse one branch point. Returns the branch result dict."""
    t_start = time.time()
    trace_dir, names, offsets, N, state_files, g_files = (
        ctx["trace_dir"], ctx["names"], ctx["offsets"], ctx["N"],
        ctx["state_files"], ctx["g_files"],
    )
    replay_dtype = torch.float64 if args.replay_dtype == "fp64" else torch.float32
    state_file = state_files[t0]
    meta, params, w_parts, m_parts, v_parts, g_parts = build_branch_state(
        state_file, names, replay_dtype, catalog=ctx.get("catalog"),
    )
    window_steps = window_steps_for(g_files, t0, args.window)
    if not window_steps:
        raise ValueError(
            f"branch t0={t0}: no g files in window (need contiguous "
            f"step_{t0:06d}_g.pt onward)"
        )
    W = len(window_steps)
    if args.verbose:
        print(f"[t0={t0}] window steps {window_steps} (W={W})")

    eff_horizons = sorted({max(1, min(h, W)) for h in args.horizons})
    horiz_steps = {str(h): max(1, min(h, W)) for h in args.horizons}

    # ---- scores ---------------------------------------------------------
    allowed = list(all_score_names())
    if args.score_names:
        unknown = [s for s in args.score_names if s not in allowed]
        if unknown:
            raise ValueError(
                f"--score-names {unknown} not a registered score "
                f"(allowed: {allowed})")
        score_names = [s for s in allowed if s in args.score_names]
    else:
        score_names = allowed
    score_flats = compute_scores_flat(
        w_parts, m_parts, v_parts, g_parts, offsets, N, meta,
        args.chunk_mb, args.device, score_names=score_names,
        update_includes_wd=False,
        hp_parts={p["name"]: (p["b1"], p["b2"], p["eps"], p["wd"])
                  for p in params},
    )
    score_stats = {s: _score_stats(score_flats[s]) for s in score_names}

    # ---- dense replay -> label / denominators / snapshots / recon ----------
    # (--dense-cache: the dense pass is the only whole-window read; on a hit
    # its artifacts load from one file and the dense working states are never
    # created — the sparse replays below only need w0/m0/v0.)
    use_dev = args.device != "cpu"
    upcast_bf16 = args.trace_dtype == "bf16"
    norm_fp32 = args.norm_dtype == "fp32"
    cached = None
    if args.dense_cache:
        ident = _dense_cache_ident(trace_dir, t0, window_steps, eff_horizons,
                                   replay_dtype, args.trace_dtype,
                                   "dev" if use_dev else "cpu")
        cached = load_dense_cache(args.dense_cache, ident)
    if cached is not None:
        label_total = cached["label_total"]
        den_at = cached["den_at"]
        recon_steps = cached["recon_steps"]
        snap_d = cached["snap_d"]
        label_flat = cached["label_flat"]
        if args.verbose:
            print(f"  [t0={t0}] dense cache hit: {cached['path']}", flush=True)
    else:
        if not use_dev:
            for prm in params:
                prm["w"] = prm["w0"].clone()
                prm["m"] = prm["m0"].clone()
                prm["v"] = prm["v0"].clone()
                prm["label"] = torch.zeros(prm["n"], dtype=replay_dtype)
            label_total, den_at, recon_steps, snap_d = dense_replay(
                params, g_files, window_steps, meta, t0, eff_horizons,
                replay_dtype, args.chunk_mb, args.device,
                upcast_bf16=upcast_bf16,
            )
        else:
            label_total, den_at, recon_steps, snap_d = dense_replay_dev(
                params, g_files, window_steps, meta, t0, eff_horizons,
                replay_dtype, args.chunk_mb, args.device,
                upcast_bf16=upcast_bf16,
            )
        if args.verbose:
            print(f"  [t0={t0}] dense replay done; den={den_at}", flush=True)
        # label flat (fp32) used by the ranking metrics
        label_flat = torch.cat([prm["label"].float() for prm in params])
        if args.dense_cache:
            cache_path = save_dense_cache(
                args.dense_cache, ident, label_total, den_at, recon_steps,
                snap_d, label_flat)
            if args.verbose:
                print(f"  [t0={t0}] dense cache written: {cache_path}",
                      flush=True)
    label_stats = _score_stats(label_flat)
    # the dense running states are no longer needed after den/label/snapshots
    # are in; free them so each sparse replay (which clones its own sparse
    # m/v/w states) does not stack on top of the dense footprints.
    for prm in params:
        for key in ("w", "m", "v", "label"):
            prm.pop(key, None)

    # ---- per-ratio / per-score metrics ----------------------------------
    per_ratio = {}
    for r in args.ratios:
        if r <= 0.0 or r > 100.0:
            raise ValueError(f"ratio {r} out of (0,100]")
        k = max(1, int(round(r / 100.0 * N)))
        label_sel, _ = select_topk_flat(label_flat, k)
        total_label = label_total
        ratio_res = {"k": k, "scores": {}}
        idcg = idcg_for(label_flat, label_sel, k)
        for sname in score_names:
            sc_flat = score_flats[sname]
            sel, thresh = select_topk_flat(sc_flat, k)
            if use_dev:
                num_at = sparse_replay_dev(
                    params, g_files, window_steps, meta, t0, eff_horizons, sel,
                    snap_d, replay_dtype, args.chunk_mb, args.device,
                    upcast_bf16=upcast_bf16, norm_fp32=norm_fp32,
                )
            else:
                num_at = sparse_replay(
                    params, g_files, window_steps, meta, t0, eff_horizons, sel,
                    snap_d, replay_dtype, args.chunk_mb, args.device,
                    upcast_bf16=upcast_bf16, norm_fp32=norm_fp32,
                )
            if args.verbose:
                print(f"  [t0={t0}] {sname}@{r}% sparse replay done "
                      f"(thresh={thresh:.3e})", flush=True)
            distortion = {}
            for h in args.horizons:
                hh = max(1, min(h, W))
                den = den_at.get(hh)
                num = num_at.get(hh)
                distortion[str(h)] = (
                    num / den if (den is not None and num is not None and den > 0)
                    else None
                )
            mets = ranking_metrics(
                sc_flat, label_flat, sel, label_sel, k,
                float(score_stats[sname]["sum"]), total_label, idcg,
            )
            metrics = {
                "threshold": thresh,
                "distortion": distortion,
                **mets,
            }
            ratio_res["scores"][sname] = metrics
            del sel, num_at
        per_ratio[str(r)] = ratio_res
        del label_sel

    branch = {
        "t0": t0,
        "state_absent": bool(meta.get("state_absent", False)),
        "window_len": W,
        "window_steps": window_steps,
        "horizon_steps": horiz_steps,
        "meta": {
            "lr": scalar_list(meta.get("lr", [])),
            "grad_norm": float(meta.get("grad_norm", float("nan"))),
            "clip_grad": float(meta.get("clip_grad", 0.0)),
            "weight_decay": float(meta.get("weight_decay", 0.0)),
            "adam_beta1": float(meta.get("adam_beta1", 0.9)),
            "adam_beta2": float(meta.get("adam_beta2", 0.999)),
            "adam_eps": float(meta.get("adam_eps", 1e-8)),
        },
        "replay": {
            "dtype": args.replay_dtype,
            "trace_dtype": args.trace_dtype,
            "dense_cache_hit": (
                (cached is not None) if args.dense_cache else None),
            "per_step": recon_steps,
            "den_norm_at_horizon": {
                str(h): den_at.get(h) for h in eff_horizons if den_at.get(h) is not None
            },
            "label_total": label_total,
            "label_stats": label_stats,
        },
        "scores": score_stats,
        "per_ratio": per_ratio,
    }
    elapsed = time.time() - t_start
    if args.verbose:
        print(f"[t0={t0}] branch processed in {elapsed:.1f}s")
    return branch


# ----------------------------------------------------------------------------
# replay self-check (--self-check FROM TO)
# ----------------------------------------------------------------------------

def run_self_check(ctx, from_step, to_step, args):
    """Validate the dense replay machinery against ground truth: load the
    real pre-commit state at ``from_step``, replay the logged gradients of
    steps [from_step, to_step) densely (log lr / wd / clip exactly as
    traced), and compare the replayed w/m/v against the real state dump at
    ``to_step``. Returns the report section."""
    names = ctx["names"]
    state_files = ctx["state_files"]
    g_files = ctx["g_files"]
    if from_step not in state_files or to_step not in state_files:
        raise ValueError(
            f"--self-check {from_step} {to_step}: both endpoints must be "
            f"branch points with state dumps (found states: {sorted(state_files)})")
    if to_step <= from_step:
        raise ValueError("--self-check requires TO_STEP > FROM_STEP")
    steps = list(range(from_step, to_step))
    missing = [s for s in steps if s not in g_files]
    if missing:
        raise ValueError(
            f"--self-check {from_step} {to_step}: missing g dump(s) at steps {missing}")

    replay_dtype = torch.float64 if args.replay_dtype == "fp64" else torch.float32
    meta, params, w_parts, m_parts, v_parts, g_parts = build_branch_state(
        state_files[from_step], names, replay_dtype, catalog=ctx.get("catalog"))
    for prm in params:
        prm["w"] = prm["w0"].clone()
        prm["m"] = prm["m0"].clone()
        prm["v"] = prm["v0"].clone()
        prm["label"] = torch.zeros(prm["n"], dtype=replay_dtype)
    W = to_step - from_step
    label_total, den_at, recon_steps, _snap = dense_replay(
        params, g_files, steps, meta, from_step, [], replay_dtype,
        args.chunk_mb, args.device,
        upcast_bf16=(args.trace_dtype == "bf16"),
    )

    real = _load(state_files[to_step])
    real_w = real["payload"]["w"]
    real_m = real["payload"].get("m", {})
    real_v = real["payload"].get("v", {})
    per_param = {}
    sum2 = {"w": 0.0, "m": 0.0, "v": 0.0}
    den2 = {"w": 0.0, "m": 0.0, "v": 0.0}
    for prm in params:
        n = prm["name"]
        entry = {}
        re = real_w[n].reshape(-1).to(replay_dtype)
        dw = (prm["w"] - re)
        s2 = float((dw.to(torch.float64) ** 2).sum())
        d2 = float((re.to(torch.float64) ** 2).sum())
        entry["w"] = math.sqrt(s2) / math.sqrt(d2) if d2 > 0 else None
        sum2["w"] += s2
        den2["w"] += d2
        for k, pk in (("m", prm["m"]), ("v", prm["v"])):
            rk = real_m.get(n) if k == "m" else real_v.get(n)
            if rk is None:
                entry[k] = None
                continue
            rk = rk.reshape(-1).to(replay_dtype)
            dk = (pk - rk)
            s2k = float((dk.to(torch.float64) ** 2).sum())
            d2k = float((rk.to(torch.float64) ** 2).sum())
            entry[k] = math.sqrt(s2k) / math.sqrt(d2k) if d2k > 0 else None
            sum2[k] += s2k
            den2[k] += d2k
        per_param[n] = entry

    overall = {
        k: (math.sqrt(sum2[k]) / math.sqrt(den2[k]) if den2[k] > 0 else None)
        for k in ("w", "m", "v")
    }
    w_vals = [per_param[n]["w"] for n in names if per_param[n]["w"] is not None]
    res = {
        "from_step": from_step,
        "to_step": to_step,
        "replay_steps": steps,
        "steps_replayed": W,
        "trace_dtype": args.trace_dtype,
        "state_absent_from": bool(meta.get("state_absent", False)),
        "per_step": recon_steps,
        "overall_relative_l2": overall,
        "per_param_w_rel_l2": {
            "min": min(w_vals) if w_vals else None,
            "median": (sorted(w_vals)[len(w_vals) // 2] if w_vals else None),
            "max": max(w_vals) if w_vals else None,
            "params": per_param,
        },
    }
    return res


def print_self_check(res):
    ov = res["overall_relative_l2"]
    print("=" * 100)
    print(f"SELF-CHECK: dense replay from step {res['from_step']} to "
          f"{res['to_step']} ({res['steps_replayed']} commits)")
    print("=" * 100)
    print("  overall relative L2 error vs real state dump:")
    for k in ("w", "m", "v"):
        v = ov.get(k)
        print(f"    {k}: {v if v is None else f'{v:.3e}'}")
    pp = res["per_param_w_rel_l2"]
    print(f"  per-param w rel L2: min={pp['min']:.3e} "
          f"median={pp['median']:.3e} max={pp['max']:.3e}")
    mx = max(pp["params"], key=lambda n: pp["params"][n]["w"] or 0.0)
    print(f"  worst param: {mx}  w={pp['params'][mx]['w']:.3e} "
          f"m={pp['params'][mx]['m']} v={pp['params'][mx]['v']}")
    bad = [n for n in pp["params"] if (pp["params"][n].get("w") or 0.0) > 1e-3]
    print(f"  params with w rel err > 1e-3: {len(bad)} "
          f"{bad[:10]}{'...' if len(bad) > 10 else ''}")
    print("=" * 100)


# ----------------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------------

def _round(x, nd=6):
    if x is None:
        return None
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return x
        return round(x, nd)
    return x


def sanitize(obj, nd=6):
    if isinstance(obj, dict):
        return {k: sanitize(v, nd) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v, nd) for v in obj]
    if isinstance(obj, float):
        return _round(obj, nd)
    return obj


def _fmt_dist(x):
    """One distortion table cell. None (dense denominator 0, i.e. the dense
    arm did not move) prints as N/A instead of crashing the format."""
    return f"{x:>12.5f}" if x is not None else "         N/A"


def print_summary(report):
    br = report["branch_results"]
    print("=" * 100)
    print(f"SGA REPLAY ORACLE — {report['trace_dir']}")
    print("=" * 100)
    for t0k in sorted(br, key=int):
        b = br[t0k]
        score_names = list(b["scores"].keys())
        print(f"\nbranch t0={b['t0']}  (window_len={b['window_len']}, "
              f"state_absent={b['state_absent']}, dtype={b['replay']['dtype']})")
        print("  scores: " + "  ".join(
            f"{s}={b['scores'][s]['max']:.4e}(max)" for s in score_names))
        for rk in sorted(b["per_ratio"], key=float):
            pr = b["per_ratio"][rk]
            print(f"  ratio {rk}%  k={pr['k']}")
            # per-ratio entries iterate by their own keys: with a T set the
            # score names carry an @T{t} suffix and supersede score_names
            first = next(iter(pr["scores"].values()))
            hs = sorted(first["distortion"], key=lambda h: int(h))
            hdr = "    score        " + " ".join(f"d@H={h:>10s}" for h in hs) + \
                  "     ndcg@k     recall@k  miss_label miss_score"
            print(hdr)
            for s in pr["scores"]:
                met = pr["scores"][s]
                dists = " ".join(_fmt_dist(met['distortion'][h]) for h in hs)
                print(f"    {s:>12s}  {dists}   {met['ndcg_at_k']:>9.5f} "
                      f"{met['recall_at_k']:>9.5f} {met['missed_label_mass']:>9.5f} "
                      f"{met['missed_score_mass']:>10.5f}")
    print("=" * 100)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="SGA offline replay oracle (score candidates vs future "
                    "distortion & ranking metrics on recorded traces)")
    ap.add_argument("--trace-dir", required=True,
                    help="trace directory (catalog.pt + step_*.pt)")
    ap.add_argument("--branch-point", default=None,
                    help="branch step (trace numbering) or 'all' for every "
                         "step_*_state.pt in the dir")
    ap.add_argument("--self-check", nargs=2, type=int, metavar=("FROM_STEP", "TO_STEP"),
                    default=None,
                    help="replay self-check: densely replay the logged gradients "
                         "from the state dump at FROM_STEP up to TO_STEP and compare "
                         "w/m/v against the real state dump at TO_STEP")
    ap.add_argument("--ratios", default="10,20",
                    help="comma list of support sizes in percent of elements")
    ap.add_argument("--horizons", default="1,10,100",
                    help="comma list of replay horizons (clipped to window)")
    ap.add_argument("--out", default=None, help="path to write the JSON report")
    ap.add_argument("--device", default="cpu", help="cpu | cuda:N")
    ap.add_argument("--chunk-mb", type=int, default=256,
                    help="per-chunk working-set budget for large params, MB")
    ap.add_argument("--replay-dtype", default="fp64", choices=["fp64", "fp32"],
                    help="dtype for the AdamW replay arithmetic")
    ap.add_argument("--norm-dtype", default="fp64", choices=["fp64", "fp32"],
                    help="accumulation dtype of the per-step clip-norm pass "
                         "(the sparse pass-1 reductions): fp64 (default, "
                         "oracle semantics) or fp32 (per-chunk fp32 "
                         "reduction, chunk sums promoted to fp64; the "
                         "deviation is gated by the frozen fp32-norm band)")
    ap.add_argument("--trace-dtype", default="fp32", choices=["fp32", "bf16"],
                    help="g payload dtype in the trace: fp32 (original "
                         "dumps) or bf16 (screening copies built by "
                         "sga.trace_convert; payloads upcast to fp32 at "
                         "load, replay arithmetic unchanged)")
    ap.add_argument("--dense-cache", default=None, metavar="DIR",
                    help="cache directory for the dense replay artifacts "
                         "(label/den/snapshots/recon, one file per trace + "
                         "t0 + window + horizons + dtype + semantic "
                         "version); repeated runs reuse the whole dense "
                         "pass instead of re-reading the window")
    ap.add_argument("--replay-mode", default="zerofill",
                    choices=["zerofill", "cycle"],
                    help="zerofill (default): frozen mask, zero-filled q each "
                         "step (original offline semantics), replayed "
                         "in-process. cycle: residual accumulation + "
                         "full-sync fold + mask re-selection every "
                         "--cycle-len steps (offline twin of the online "
                         "cyclic semantics), always served by the fused "
                         "single-pass engine — requires --dense-cache (a "
                         "first run populates it automatically)")
    ap.add_argument("--cycle-len", type=int, default=None,
                    help="T: full-sync / mask re-selection interval for "
                         "--replay-mode cycle (window boundary si=0 is always "
                         "a full-sync step); single-T spelling of "
                         "--cycle-lens, mutually exclusive with it")
    ap.add_argument("--cycle-lens", default=None,
                    help="comma set of T intervals for --replay-mode cycle "
                         "(e.g. 5,20,100): every (score, ratio, T) triple "
                         "becomes a replay config and mixed-T groups share "
                         "one window pass; mutually exclusive with "
                         "--cycle-len (default: 20, as a one-element set)")
    ap.add_argument("--fused", action="store_true",
                    help="(deprecated no-op: --replay-mode cycle now always "
                         "runs the fused engine; kept so existing command "
                         "lines keep working)")
    ap.add_argument("--fused-shards", type=int, default=8,
                    help="shard process count for --replay-mode cycle "
                         "(fused engine; shard r uses cuda:r; the window "
                         "read, the streaming AdamW state and the top-k "
                         "select are split across shards)")
    ap.add_argument("--fused-max-configs", type=int, default=12,
                    help="arms (score x ratio) advanced per fused window "
                         "pass in --replay-mode cycle (config group size; "
                         "state memory per group scales with this)")
    ap.add_argument("--fused-state-budget-mb", type=float, default=0.0,
                    help="resident-state byte budget per shard (MiB) for "
                         "--replay-mode cycle: the config group size is "
                         "derived as floor(budget / per-config state bytes) "
                         "instead of --fused-max-configs, which this "
                         "overrides when > 0. Accounting (the plan-level "
                         "budget table): {w,m,v,res} x replay-dtype "
                         "itemsize over the widest shard (ceil(N/world)); "
                         "the bool selection mask, the --chunk-mb working "
                         "set and g-residency buffers sit OUTSIDE the "
                         "budget. llama-1B at 8 shards costs ~2359 MiB per "
                         "config in fp32, so 30720 derives R=13. 0 = off "
                         "(the count knob rules)")
    ap.add_argument("--nodes", default=None,
                    help="comma list of node addresses for a multi-node "
                         "fused run (sga.replay_launch; this process must "
                         "run on the FIRST node and share the other nodes' "
                         "NFS path view; shard r -> node r // (shards/nodes), "
                         "device cuda:(r %% (shards/nodes)); requires "
                         "--replay-mode cycle; nodes are reached as "
                         "root@node over ssh port 2222, and "
                         "SGA_LAUNCH_EXECUTOR=local overrides that with an "
                         "all-local single-machine simulation — the "
                         "executor actually used is printed in the launch "
                         "banner)")
    ap.add_argument("--ssh-key", default=None,
                    help="cluster ssh key for --nodes launches (default: "
                         "$SGA_SSH_KEY, then the nearest workspace ancestor's "
                         "cluster_ssh_key)")
    ap.add_argument("--preflight-timeout", type=float, default=600.0,
                    help="bounded rendezvous preflight deadline in seconds "
                         "for --nodes runs (loud failure on expiry)")
    ap.add_argument("--run-deadline", type=float, default=0.0,
                    help="optional wall-clock deadline in seconds for the "
                         "--nodes run phase itself: on expiry every "
                         "node-worker is killed and the launch fails loudly "
                         "with the per-node log tails (0 disables the "
                         "bound — the default, since legitimate replays "
                         "run for hours; a shard hung outside a collective "
                         "is otherwise unbounded)")
    ap.add_argument("--launch-log-dir", default=None,
                    help="directory for per-node launch logs of a --nodes "
                         "run (default: a timestamped dir next to the run's "
                         "temp files)")
    ap.add_argument("--fused-g-residency", action="store_true",
                    help="upload each step's local g slices to the device "
                         "once and share the resident copy across every "
                         "config and all three streaming passes instead of "
                         "re-uploading per config per pass (bitwise "
                         "identical arithmetic; the pinned host buffers "
                         "plus non_blocking keep the HOST unblocked, the "
                         "H2D itself still serializes on the compute "
                         "stream). Memory: "
                         "residency holds full per-entry g slices on "
                         "device per step (independent of the --chunk-mb "
                         "working-set budget; the per-step divide "
                         "transiently doubles that footprint) and on CUDA "
                         "pins ~depth+1 steps of shard g slices in "
                         "page-locked host memory (sizeable at production "
                         "shard sizes)")
    ap.add_argument("--fused-dev-reduce", action="store_true",
                    help="accumulate the pass-1 clip norms and pass-3 "
                         "horizon numerators in per-config device scalars "
                         "and fetch them once per pass instead of a host "
                         "sync per chunk per config (same addends in the "
                         "same order, so bitwise identical; at most one "
                         "host sync per pass per step — the clip-norm "
                         "fetch every step, plus the horizon fetch only "
                         "on steps where a horizon lands)")
    ap.add_argument("--score-names", default=None,
                    help="comma list restricting the score set (validated "
                         "against the framework score registry). Default: all "
                         "registered scores")
    ap.add_argument("--window", type=int, default=100,
                    help="trace window length (matches the collector's "
                         "sga.trace.window)")
    ap.add_argument("--seed", type=int, default=1234, help="RNG seed")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    # Option-value gate before any work: every invalid value is rejected with
    # its option named, instead of surfacing as an incidental deep exception
    # (si % T_c) or a silently empty report (',' branch specs).
    # The cycle knob resolves to the per-config T list here: --cycle-len N is
    # the one-element list, the two spellings are mutually exclusive, and the
    # set is sorted/deduped so the config enumeration is deterministic.
    if args.cycle_len is not None and args.cycle_lens is not None:
        raise ValueError(
            "--cycle-len and --cycle-lens are mutually exclusive: pass the "
            "T set once (--cycle-len N or --cycle-lens T1,T2,...)")
    if args.cycle_lens is not None:
        toks = [t for t in args.cycle_lens.replace(" ", "").split(",") if t]
        if not toks:
            raise ValueError(
                f"--cycle-lens {args.cycle_lens!r} names no cycle length")
        lens = []
        for tok in toks:
            try:
                t = int(tok)
            except ValueError:
                raise ValueError(
                    f"--cycle-lens values must be integers >= 1, "
                    f"got {tok!r}") from None
            if t < 1:
                raise ValueError(
                    f"--cycle-lens values must be >= 1, got {tok!r}")
            lens.append(t)
        args.cycle_lens = sorted(set(lens))
    else:
        if args.cycle_len is None:
            args.cycle_len = 20  # default T, kept for compatibility
        if args.cycle_len < 1:
            # cycle mode needs a positive full-sync interval; under zerofill
            # the knob is unused but still echoed into the report config, so
            # the gate covers both modes
            raise ValueError(f"--cycle-len must be >= 1, got {args.cycle_len}")
        args.cycle_lens = [args.cycle_len]
    if args.replay_mode == "cycle" and args.self_check is not None:
        # cycle dispatches to run_fused_branch, which the self-check path
        # never reaches — reject the combination by name instead of
        # silently running the dense self-check
        raise ValueError(
            "--replay-mode cycle cannot be combined with --self-check")
    if args.replay_mode == "cycle" and not args.dense_cache:
        raise ValueError(
            "--replay-mode cycle requires --dense-cache (the classic "
            "in-process cycle replay that ran without a cache was removed; "
            "the fused engine consumes the dense artifacts from the cache "
            "and populates it on the first run)")
    if args.fused and args.replay_mode != "cycle":
        raise ValueError(
            "--fused is deprecated (cycle mode now always runs the fused "
            "engine) and --replay-mode zerofill has no fused form")
    if args.nodes and args.replay_mode != "cycle":
        raise ValueError(
            "--nodes is only supported with --replay-mode cycle (the "
            "multi-node launcher serves the fused engine only)")
    if args.nodes is not None:
        args.nodes = [x.strip() for x in args.nodes.split(",") if x.strip()]
        if not args.nodes:
            # ',' survives the truthy gate above; an explicitly requested
            # multi-node launch that names no node must fail loudly instead
            # of silently degrading to a single-node run
            raise ValueError("--nodes names no node")
        if args.preflight_timeout <= 0:
            raise ValueError(
                f"--preflight-timeout must be > 0, "
                f"got {args.preflight_timeout}")
        if args.run_deadline < 0:
            raise ValueError(
                f"--run-deadline must be >= 0 (0 disables the bound), "
                f"got {args.run_deadline}")
    if args.chunk_mb < 1:
        raise ValueError(f"--chunk-mb must be >= 1, got {args.chunk_mb}")
    if args.window < 1:
        raise ValueError(f"--window must be >= 1, got {args.window}")
    # fused-only knobs (shard count / configs per fused window pass):
    # unused under zerofill but still echoed into the report config, so an
    # invalid value is rejected here with the option named;
    # run_fused_branch re-checks both for direct programmatic callers
    if args.fused_shards < 1:
        raise ValueError(
            f"--fused-shards must be >= 1, got {args.fused_shards}")
    if args.fused_max_configs < 1:
        raise ValueError(
            f"--fused-max-configs must be >= 1, got {args.fused_max_configs}")
    if not math.isfinite(args.fused_state_budget_mb) \
            or args.fused_state_budget_mb < 0:
        raise ValueError(
            "--fused-state-budget-mb must be a finite value >= 0, got "
            f"{args.fused_state_budget_mb}")
    for tok in args.ratios.split(","):
        if tok.strip():
            r = float(tok)
            if not (math.isfinite(r) and 0.0 < r <= 100.0):
                raise ValueError(
                    f"--ratios value must be finite in (0, 100], got {tok!r}")
    for tok in args.horizons.split(","):
        if tok.strip() and int(tok) < 1:
            raise ValueError(f"--horizons values must be >= 1, got {tok!r}")
    if (args.branch_point is not None and args.self_check is None
            and args.branch_point.strip().lower() != "all"
            and not [t for t in args.branch_point.replace(" ", "").split(",")
                     if t]):
        raise ValueError(
            f"--branch-point {args.branch_point!r} names no branch step")
    torch.manual_seed(args.seed)
    if args.device != "cpu":
        if not args.device.startswith("cuda"):
            raise ValueError(f"--device must be 'cpu' or 'cuda:N', got {args.device}")
        if not torch.cuda.is_available():
            raise ValueError("--device cuda requested but CUDA is unavailable")
    if args.self_check is None and not args.branch_point:
        raise ValueError("need either --branch-point or --self-check")
    if args.self_check is not None and args.branch_point:
        raise ValueError("--self-check and --branch-point are mutually exclusive")
    if args.self_check is not None and args.dense_cache:
        raise ValueError(
            "--dense-cache cannot be combined with --self-check "
            "(the self-check must actually replay the window)")

    trace_dir = os.path.abspath(args.trace_dir)
    catalog, names, offsets, N, state_files, g_files = discover_trace(trace_dir)
    if args.verbose:
        print(f"trace {trace_dir}: {len(names)} params, {N} elements "
              f"(~{N*4/1e9:.2f} GB fp32)")

    ctx = {
        "trace_dir": trace_dir, "names": names, "offsets": offsets, "N": N,
        "state_files": state_files, "g_files": g_files, "catalog": catalog,
    }

    # ---- self-check mode: validate the dense replay against real state ----
    if args.self_check is not None:
        args.ratios = sorted({float(x) for x in args.ratios.split(",") if x.strip()})
        args.horizons = sorted({int(x) for x in args.horizons.split(",") if x.strip()})
        sc = run_self_check(ctx, args.self_check[0], args.self_check[1], args)
        report = {
            "trace_dir": trace_dir,
            "config": {
                "self_check": [args.self_check[0], args.self_check[1]],
                "device": args.device,
                "chunk_mb": args.chunk_mb,
                "replay_dtype": args.replay_dtype,
                "seed": args.seed,
            },
            "self_check": sc,
        }
        report = sanitize(report, nd=8)
        if args.out:
            out_path = args.out if os.path.isabs(args.out) else os.path.abspath(args.out)
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(report, f, indent=2)
            if args.verbose:
                print(f"report written to {out_path}")
        print_self_check(sc)
        return report

    args.ratios = sorted({float(x) for x in args.ratios.split(",") if x.strip()})
    args.horizons = sorted({int(x) for x in args.horizons.split(",") if x.strip()})
    if not args.ratios:
        raise ValueError("--ratios empty")
    if not args.horizons:
        raise ValueError("--horizons empty")
    if args.score_names is not None:
        args.score_names = [s for s in
                            (x.strip() for x in args.score_names.split(",")) if s]
        if not args.score_names:
            raise ValueError("--score-names empty")
    branch_pts = parse_branch_spec(args.branch_point, state_files)
    if args.verbose:
        print(f"branch points {branch_pts}")

    branch_results = {}
    for t0 in branch_pts:
        if args.replay_mode == "cycle":
            if args.nodes:
                from sga.replay_launch import run_fused_branch_multinode
                branch_results[str(t0)] = run_fused_branch_multinode(
                    args, ctx, t0, args.nodes, ssh_key=args.ssh_key)
            else:
                from sga.replay_fused import run_fused_branch
                branch_results[str(t0)] = run_fused_branch(args, ctx, t0)
        else:
            branch_results[str(t0)] = process_branch(t0, ctx, args)

    report = {
        "trace_dir": trace_dir,
        "config": {
            "branch_points": branch_pts,
            "ratios": args.ratios,
            "horizons": args.horizons,
            "device": args.device,
            "chunk_mb": args.chunk_mb,
            "replay_dtype": args.replay_dtype,
            "trace_dtype": args.trace_dtype,
            "norm_dtype": args.norm_dtype,
            "dense_cache": args.dense_cache,
            "replay_mode": args.replay_mode,
            "cycle_len": args.cycle_len,
            "cycle_lens": args.cycle_lens,
            "fused": args.fused,
            "fused_effective": args.fused or args.replay_mode == "cycle",
            "fused_shards": args.fused_shards,
            "fused_max_configs": args.fused_max_configs,
            "nodes": args.nodes,
            "fused_g_residency": args.fused_g_residency,
            "fused_dev_reduce": args.fused_dev_reduce,
            "score_names": args.score_names,
            "window": args.window,
            "seed": args.seed,
        },
        "branch_results": branch_results,
    }
    # conditional echo: the budget-off report must stay byte-identical to
    # budget-off artifacts of this version (cycle_lens above is echoed
    # unconditionally, so pre-knob artifacts already differ at byte level),
    # so this key only exists when the knob is on
    if args.fused_state_budget_mb > 0:
        report["config"]["fused_state_budget_mb"] = \
            args.fused_state_budget_mb
    report = sanitize(report, nd=8)

    if args.out:
        out_path = args.out if os.path.isabs(args.out) else os.path.abspath(args.out)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        if args.verbose:
            print(f"report written to {out_path}")

    print_summary(report)
    return report


if __name__ == "__main__":
    main()
