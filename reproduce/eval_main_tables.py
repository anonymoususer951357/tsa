#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from eval_archives import job_id, patch_acc, patch_tsim, read_done  # noqa: E402

DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "reproduce" / "data"))
WEIGHTS = ROOT / "reproduce" / "weights"
OUT_JSON = ROOT / "reproduce" / "main_table_results.txt"

DATASETS = ["movi_c", "movi_e", "ytvis_hq", "ovis"]

METHOD_META = {
    "videosaur": ("videosaur", "VideoSAUR", 25.1, 344),
    "slotcontrast": ("slotcontrast", "SlotContrast", 31.4, 295),
    "randsfq-tsim": ("randsfq_tsim", r"RandSF.Q\textsubscript{tsim}", 34.1, 411),
    "randsfq": ("randsfq_ssc", r"RandSF.Q\textsubscript{ssc}", 34.1, 411),
    "smoothsa": ("smoothsa", "SmoothSA", 35.3, 373),
    "tsa": ("tsa", r"\textbf{TSA (ours)}", 34.2, 359),
}
METHOD_ORDER = list(METHOD_META.keys())
METRIC_KEYS = ["ari_fg", "mbo", "hota", "idf1"]


def _resolve_patch(tags):
    fns = []
    for t in tags:
        if t == "acc":
            fns.append(patch_acc)
        elif t == "tsim":
            fns.append(patch_tsim)
        else:
            raise KeyError(t)
    if not fns:
        return None

    def _patch(cfg):
        for fn in fns:
            cfg = fn(cfg)
        return cfg

    _patch.patch_tag = ",".join(tags)
    return _patch


def _cfg_and_patch(method: str, dataset: str):
    tags = []
    if dataset == "ovis":
        tags.append("acc")

    if method == "tsa":
        cfg = ROOT / "config-tsa" / f"tsa-{dataset}.py"
    elif method == "smoothsa":
        if dataset in ("movi_c", "movi_e"):
            cfg = ROOT / "config-smoothsa" / f"smoothsav_c-{dataset}.py"
        else:
            cfg = ROOT / "config-smoothsa" / f"smoothsav_r-{dataset}.py"
    elif method == "slotcontrast":
        if dataset in ("movi_c", "movi_e"):
            cfg = ROOT / "config-slotcontrast" / f"slotcontrast_c-{dataset}.py"
        else:
            cfg = ROOT / "config-slotcontrast" / f"slotcontrast_r-{dataset}.py"
    elif method == "videosaur":
        if dataset in ("movi_c", "movi_e"):
            cfg = ROOT / "config-videosaur" / f"videosaur_c-{dataset}.py"
        else:
            cfg = ROOT / "config-videosaur" / f"videosaur_r-{dataset}.py"
    elif method == "randsfq":
        if dataset in ("movi_c", "movi_e"):
            cfg = ROOT / "config-randsfq" / f"randsfq_c-{dataset}.py"
        elif dataset == "ytvis_hq":
            cfg = ROOT / "config-randsfq" / "randsfq_r-ytvis_hq.py"
        else:
            cfg = ROOT / "config-randsfq" / "randsfq_r-ovis.py"
    elif method == "randsfq-tsim":
        if dataset in ("movi_c", "movi_e"):
            cfg = ROOT / "config-randsfq-tsim" / f"randsfq_c-{dataset}-tsim.py"
        elif dataset == "ytvis_hq":
            cfg = ROOT / "config-randsfq-tsim" / "randsfq_r-ytvis_hq-tsim.py"
        else:
            cfg = ROOT / "config-randsfq-tsim" / "randsfq_r-ovis-tsim.py"
    else:
        raise KeyError(method)

    return cfg, _resolve_patch(tags)


def all_jobs():
    jobs = []
    for method in METHOD_ORDER:
        wdir = WEIGHTS / METHOD_META[method][0]
        for dataset in DATASETS:
            cfg, patch = _cfg_and_patch(method, dataset)
            assert cfg.is_file(), f"missing config: {cfg}"
            for seed in (1, 2, 3):
                ckpt = wdir / dataset / f"weight{seed}.pth"
                assert ckpt.is_file(), f"missing checkpoint: {ckpt}"
                jobs.append(
                    dict(
                        method=method,
                        dataset=dataset,
                        seed=seed,
                        cfg_file=cfg,
                        ckpt_file=ckpt,
                        patch=patch,
                        patch_tag=getattr(patch, "patch_tag", "none") if patch else "none",
                    )
                )
    assert len(jobs) == 72, len(jobs)
    return jobs


def mean_std(vals):
    a = np.asarray(vals, dtype=np.float64) * 100.0
    return float(a.mean()), float(a.std(ddof=0))


def group_metrics(done):
    g = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for rec in done.values():
        for k in METRIC_KEYS:
            if k in rec["metrics"]:
                g[rec["method"]][rec["dataset"]][k].append(rec["metrics"][k])
    return g


def write_summary(done, path: Path):
    lines = []
    for rec in sorted(done.values(), key=lambda r: (r["method"], r["dataset"], r["seed"])):
        lines.append("RESULT " + json.dumps(rec))
    lines += ["", "=" * 100, "SUMMARY mean±std (x100)", "=" * 100]
    header = f"{'method':<14}{'dataset':<10}{'n':>3}" + "".join(f"{k:>14}" for k in METRIC_KEYS)
    lines.append(header)
    stats = group_metrics(done)
    for ds in DATASETS:
        for method in METHOD_ORDER:
            if ds not in stats[method]:
                continue
            n = max((len(stats[method][ds][k]) for k in METRIC_KEYS), default=0)
            cells = []
            for k in METRIC_KEYS:
                vals = stats[method][ds].get(k, [])
                if not vals:
                    cells.append(f"{'—':>14}")
                else:
                    m, s = mean_std(vals)
                    cells.append(f"{m:6.2f}±{s:5.2f}")
            lines.append(f"{method:<14}{ds:<10}{n:>3}" + "".join(f"{c:>14}" for c in cells))
        lines.append("")
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}")


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--list_only", action="store_true")
    p.add_argument("--summary_only", action="store_true")
    args = p.parse_args()

    jobs = all_jobs()
    done = read_done(OUT_JSON)
    print(f"jobs={len(jobs)} done={sum(1 for j in jobs if job_id(j) in done)} data_dir={DATA_DIR}")
    for j in jobs:
        mark = "done" if job_id(j) in done else "TODO"
        print(f"  [{mark}] {job_id(j)}")

    if args.list_only:
        return
    if args.summary_only:
        want = {job_id(j) for j in jobs}
        done = {k: v for k, v in done.items() if k in want}
        assert len(done) == 72, f"need 72 results for summary, have {len(done)}"
        write_summary(done, OUT_JSON)
        return

    raise SystemExit("Use scripts/run_main_eval.sh to run the full eval on GPUs.")


if __name__ == "__main__":
    main()
