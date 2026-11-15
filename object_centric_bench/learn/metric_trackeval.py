import importlib.util
import os
import sys
import types

import numpy as np
import torch as pt

from .metric import ARI, Metric

for _attr, _builtin in (("float", float), ("int", int), ("bool", bool)):
    if not hasattr(np, _attr):
        setattr(np, _attr, _builtin)

_TRACKEVAL_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "TrackEval")
)
_PKG_DIR = os.path.join(_TRACKEVAL_ROOT, "trackeval")
_METRICS_DIR = os.path.join(_PKG_DIR, "metrics")

def _stub_pkg(name: str, path: str) -> types.ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    pkg = types.ModuleType(name)
    pkg.__path__ = [path]
    sys.modules[name] = pkg
    return pkg

def _load_module(name: str, file_path: str) -> types.ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load TrackEval module {name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

_stub_pkg("trackeval", _PKG_DIR)
_load_module("trackeval.utils", os.path.join(_PKG_DIR, "utils.py"))
_load_module("trackeval._timing", os.path.join(_PKG_DIR, "_timing.py"))
_stub_pkg("trackeval.metrics", _METRICS_DIR)
_load_module(
    "trackeval.metrics._base_metric", os.path.join(_METRICS_DIR, "_base_metric.py")
)
_TE_HOTA = _load_module(
    "trackeval.metrics.hota", os.path.join(_METRICS_DIR, "hota.py")
).HOTA
_TE_IDENTITY = _load_module(
    "trackeval.metrics.identity", os.path.join(_METRICS_DIR, "identity.py")
).Identity

def _drop_gt_channels(target_seq: pt.Tensor, skip) -> pt.Tensor:

    if not skip:
        return target_seq
    G = target_seq.size(-1)
    keep = pt.ones(G, dtype=pt.bool, device=target_seq.device)
    for c in skip:
        if 0 <= int(c) < G:
            keep[int(c)] = False
    return target_seq[..., keep]

def _build_trackeval_data(
    input_seq: pt.Tensor,
    target_seq: pt.Tensor,
    skip_gt=(),
) -> dict:

    assert input_seq.ndim == target_seq.ndim == 3
    target_seq = _drop_gt_channels(target_seq, skip_gt)
    T = int(input_seq.size(0))

    raw_gt_ids: list = []
    raw_tracker_ids: list = []
    similarity_scores: list = []
    num_gt_dets = 0
    num_tracker_dets = 0

    for t in range(T):
        pred_t = input_seq[t].float()
        gt_t = target_seq[t].float()

        gt_present = (gt_t.sum(0) > 0).cpu().numpy()
        pred_present = (pred_t.sum(0) > 0).cpu().numpy()
        gt_ids_t = np.where(gt_present)[0].astype(np.int64)
        tracker_ids_t = np.where(pred_present)[0].astype(np.int64)

        num_gt_dets += int(gt_ids_t.size)
        num_tracker_dets += int(tracker_ids_t.size)

        if gt_ids_t.size and tracker_ids_t.size:
            gt_masks = gt_t[:, gt_ids_t].T
            pred_masks = pred_t[:, tracker_ids_t].T
            inter = (gt_masks @ pred_masks.T).cpu().numpy()
            gt_area = gt_masks.sum(1).cpu().numpy()[:, None]
            pred_area = pred_masks.sum(1).cpu().numpy()[None, :]
            union = gt_area + pred_area - inter
            sim = inter / np.maximum(union, 1e-12)
        else:
            sim = np.zeros((gt_ids_t.size, tracker_ids_t.size), dtype=np.float64)

        raw_gt_ids.append(gt_ids_t)
        raw_tracker_ids.append(tracker_ids_t)
        similarity_scores.append(sim)

    if num_gt_dets > 0:
        unique_gt_ids = np.unique(np.concatenate(raw_gt_ids))
        gt_id_map = -np.ones(int(unique_gt_ids.max()) + 1, dtype=np.int64)
        gt_id_map[unique_gt_ids] = np.arange(unique_gt_ids.size, dtype=np.int64)
        gt_ids = [
            gt_id_map[ids].astype(np.int64) if ids.size else ids
            for ids in raw_gt_ids
        ]
        num_gt_ids = int(unique_gt_ids.size)
    else:
        gt_ids = raw_gt_ids
        num_gt_ids = 0

    if num_tracker_dets > 0:
        unique_tracker_ids = np.unique(np.concatenate(raw_tracker_ids))
        tracker_id_map = -np.ones(
            int(unique_tracker_ids.max()) + 1, dtype=np.int64
        )
        tracker_id_map[unique_tracker_ids] = np.arange(
            unique_tracker_ids.size, dtype=np.int64
        )
        tracker_ids = [
            tracker_id_map[ids].astype(np.int64) if ids.size else ids
            for ids in raw_tracker_ids
        ]
        num_tracker_ids = int(unique_tracker_ids.size)
    else:
        tracker_ids = raw_tracker_ids
        num_tracker_ids = 0

    return {
        "gt_ids": gt_ids,
        "tracker_ids": tracker_ids,
        "similarity_scores": similarity_scores,
        "num_gt_ids": num_gt_ids,
        "num_tracker_ids": num_tracker_ids,
        "num_gt_dets": num_gt_dets,
        "num_tracker_dets": num_tracker_dets,
        "num_timesteps": T,
    }

def _to_scalar(value) -> float:

    if isinstance(value, np.ndarray):
        if value.size == 0:
            return 0.0
        return float(np.mean(value))
    return float(value)

class _TrackEvalMetricBase(Metric):

    _backend_cls = None
    SUPPORTED_FIELDS: set = set()

    def __init__(self, field: str, mean=(), skip=(0,), **backend_config):
        super().__init__(mean)
        if field not in self.SUPPORTED_FIELDS:
            raise ValueError(
                f"Field '{field}' not supported by {type(self).__name__}; "
                f"valid options: {sorted(self.SUPPORTED_FIELDS)}"
            )
        self.field = field
        self.skip = tuple(int(c) for c in (skip or ()))
        cfg = {"PRINT_CONFIG": False}
        cfg.update(backend_config)
        try:
            self._backend = self._backend_cls(config=cfg)
        except TypeError:

            self._backend = self._backend_cls()

    def compute(self, input, target):

        assert input.ndim == target.ndim == 4

        if self.skip:
            skip_idx = pt.from_numpy(np.asarray(self.skip, dtype=np.int64)).to(
                target.device
            )
            target_for_valid = ARI.skip_segment(target.flatten(1, 2), skip_idx)
        else:
            target_for_valid = target.flatten(1, 2)
        valid = ARI.find_valid(target_for_valid)

        scores = []
        for b in range(input.size(0)):
            data = _build_trackeval_data(input[b], target[b], skip_gt=self.skip)
            res = self._backend.eval_sequence(data)
            scores.append(_to_scalar(res[self.field]))
        return (
            pt.tensor(scores, dtype=pt.float32, device=input.device),
            valid,
        )

class HOTATrackEval(_TrackEvalMetricBase):

    _backend_cls = _TE_HOTA
    SUPPORTED_FIELDS = {
        "HOTA", "DetA", "AssA",
        "DetRe", "DetPr", "AssRe", "AssPr",
        "LocA", "OWTA",
        "HOTA_TP", "HOTA_FN", "HOTA_FP",
        "HOTA(0)", "LocA(0)", "HOTALocA(0)",
    }

    def __init__(self, field: str = "HOTA", skip=(0,), mean=()):
        super().__init__(field=field, mean=mean, skip=skip)

class IDF1TrackEval(_TrackEvalMetricBase):

    _backend_cls = _TE_IDENTITY
    SUPPORTED_FIELDS = {"IDTP", "IDFN", "IDFP", "IDR", "IDP", "IDF1"}

    def __init__(
        self, threshold: float = 0.5, field: str = "IDF1", skip=(0,), mean=()
    ):
        super().__init__(
            field=field, mean=mean, skip=skip, THRESHOLD=float(threshold)
        )
