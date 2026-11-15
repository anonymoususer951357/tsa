from pathlib import Path
import json
import random

from einops import rearrange
import numpy as np
import torch as pt

from eval import val_epoch
from object_centric_bench.datum import DataLoader, Lambda
from object_centric_bench.learn import ARI, HOTATrackEval, IDF1TrackEval, MetricWrap, mBO, mIoU
from object_centric_bench.model import Linear, ModelWrap
from object_centric_bench.util import Compose, Config, build_from_config


def patch_tsim(cfg):
    vfm_dim = cfg.vfm_dim
    hw = cfg.resolut1[0] * cfg.resolut1[1]
    cfg.model["decode"]["readout"] = dict(
        type=Linear, in_features=vfm_dim, out_features=vfm_dim + hw
    )
    recon = cfg.loss_fn_v["recon"]
    recon["transform"] = dict(
        type=Compose,
        transforms=[
            dict(type=Lambda, ikeys=[["input"]], func=lambda _: _[:, :, :vfm_dim]),
            recon["transform"],
        ],
    )
    return cfg


def patch_acc(cfg):
    acc_dict = dict(
        map=dict(input="output.segment", target="batch.segment"),
        transform=dict(
            type=Lambda,
            ikeys=[["input", "target"]],
            func=lambda _: rearrange(_, "b t h w s -> b (t h w) s"),
        ),
    )
    track_dict = dict(
        map=dict(input="output.segment", target="batch.segment"),
        transform=dict(
            type=Lambda,
            ikeys=[["input", "target"]],
            func=lambda _: rearrange(_, "b t h w s -> b t (h w) s"),
        ),
    )
    cfg.acc_fn_v = dict(
        ari=dict(metric=dict(type=ARI, skip=[]), **acc_dict),
        ari_fg=dict(metric=dict(type=ARI, skip=[0]), **acc_dict),
        mbo=dict(metric=dict(type=mBO, skip=[]), **acc_dict),
        miou=dict(metric=dict(type=mIoU, skip=[]), **acc_dict),
        idf1=dict(
            metric=dict(type=IDF1TrackEval, threshold=0.5, field="IDF1", skip=[]),
            **track_dict,
        ),
        hota=dict(metric=dict(type=HOTATrackEval, field="HOTA", skip=[]), **track_dict),
    )
    return cfg


def load_cfg(job, data_dir):
    cfg = Config.fromfile(job["cfg_file"])
    cfg.name = job["cfg_file"].name[:-3]
    cfg.dataset_t.base_dir = cfg.dataset_v.base_dir = data_dir
    if job["patch"] is not None:
        cfg = job["patch"](cfg)
    return cfg


def evaluate(job, data_dir):
    cfg = load_cfg(job, data_dir)
    dataset_v = build_from_config(cfg.dataset_v)
    dataload_v = DataLoader(
        dataset_v,
        cfg.batch_size_v,
        shuffle=False,
        num_workers=cfg.num_work,
        collate_fn=build_from_config(cfg.collate_fn_v),
        pin_memory=True,
    )
    model = ModelWrap(build_from_config(cfg.model), cfg.model_imap, cfg.model_omap)
    model.load(job["ckpt_file"], None, verbose=False)
    if cfg.freez:
        model.freez(cfg.freez, verbose=False)
    model = model.cuda()
    loss_fn_v = MetricWrap(**build_from_config(cfg.loss_fn_v))
    acc_fn_v = MetricWrap(detach=True, **build_from_config(cfg.acc_fn_v))
    cfg.callback_v = [_ for _ in cfg.callback_v if _.type.__name__ != "SaveModel"]
    for cb in cfg.callback_v:
        if cb.type.__name__ in ["AverageLog", "HandleLog"]:
            cb.log_file = None
    callback_v = build_from_config(cfg.callback_v)
    pack = val_epoch(cfg, dataload_v, model, loss_fn_v, acc_fn_v, callback_v)
    return pack.log_info


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    pt.manual_seed(seed)


def job_id(job):
    return f"{job['method']} | {job['dataset']} | {job['seed']} | {job['ckpt_file'].as_posix()}"


def read_done(out_file):
    done = {}
    out_file = Path(out_file)
    if out_file.exists():
        for line in out_file.read_text().splitlines():
            if line.startswith("RESULT "):
                rec = json.loads(line[len("RESULT "):])
                done[rec["id"]] = rec
    return done
