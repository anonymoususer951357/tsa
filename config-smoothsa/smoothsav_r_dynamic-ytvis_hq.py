import torch.nn.functional as ptnf
from einops import rearrange
from object_centric_bench.datum import (
    StridedRandomSliceSequence,
    RandomCrop,
    Resize,
    RandomFlip,
    Normalize,
    PadTo1,
    CenterCrop,
    Lambda,
    YTVIS,
    ClPadToMax1,
    DefaultCollate,
    Xywh2Ltrb,
)
from object_centric_bench.learn import *
from object_centric_bench.model import *
from object_centric_bench.util import Compose, ComposeNoStar, importlib_cfg
from object_centric_bench.util_model import interpolat_argmax_attent

max_num = 6 + 1
resolut0 = [256, 256]
resolut1 = [16, 16]
emb_dim = 256
vfm_dim = 384

total_step = 50000
val_interval = total_step // 40
batch_size_t = 64 // 4
batch_size_v = 1
num_work = 16
lr = 2e-4 / 4

burn_in_len=10
roll_out_len=5
clip_len = burn_in_len + roll_out_len

IMAGENET_MEAN = [[[123.675]], [[116.28]], [[103.53]]]
IMAGENET_STD = [[[58.395]], [[57.12]], [[57.375]]]
transform_t = [

    dict(type=RandomCrop, keys=["video", "segment"], size=None, scale=[0.75, 1]),
    dict(type=Resize, keys=["video"], size=resolut0, interp="bilinear"),
    dict(type=Resize, keys=["segment"], size=resolut0, interp="nearest-exact", c=0),
    dict(type=RandomFlip, keys=["video", "segment"], dims=[-1], p=0.5),
    dict(type=Normalize, keys=["video"], mean=[IMAGENET_MEAN], std=[IMAGENET_STD]),
    dict(type=PadTo1, keys=["video", "segment", "clazz"], dim=0, size=clip_len),
]
transform_v = [
    dict(type=CenterCrop, keys=["video", "segment"], size=None),
    dict(type=Resize, keys=["video"], size=resolut0, interp="bilinear"),
    dict(type=Resize, keys=["segment"], size=resolut0, interp="nearest-exact", c=0),
    dict(type=Normalize, keys=["video"], mean=[IMAGENET_MEAN], std=[IMAGENET_STD]),
    dict(type=PadTo1, keys=["video", "segment", "clazz"], dim=0, size=clip_len),
]
dataset_t = dict(
    type=YTVIS,
    data_file="ytvis_hq/train.lmdb",
    extra_keys=["segment", "bbox", "clazz"],
    transform0=dict(
        type=StridedRandomSliceSequence,
        keys=["video", "segment", "clazz"],
        size=clip_len,
    ),
    transform=dict(type=Compose, transforms=transform_t),
    base_dir=...,
    ts=30,
)
dataset_v = dict(
    type=YTVIS,
    data_file="ytvis_hq/val.lmdb",
    extra_keys=["segment", "bbox", "clazz"],
    transform0=dict(
        type=StridedRandomSliceSequence,
        keys=["video", "segment", "clazz"],
        size=clip_len,
    ),
    transform=dict(type=Compose, transforms=transform_v),
    base_dir=...,
)
collate_fn_t = dict(
    type=ComposeNoStar,
    transforms=[
        dict(type=ClPadToMax1, keys=["segment", "bbox", "clazz"], dims=[3, 1, 1]),
        dict(type=DefaultCollate),
    ],
)
collate_fn_v = collate_fn_t

cfg_dict = importlib_cfg(__file__.replace("_dynamic", ""))
discov = cfg_dict["model"]

model = dict(
    type=ObjDynamicPredictor,
    discov=discov,
    dynamic = dict(
        type=SlotFormer,
        num_slots=max_num,
        slot_size=emb_dim,
        history_len=burn_in_len,
        t_pe="sin",
        d_model=128,
        num_layers=4,
        ffn_dim=512,
        dropout=0.1,
    ),

    feature_idx=0,
    slotz_idx=2,
    attentd_idx=5,
)
model_imap = dict(input="batch.video")
model_omap = [
    "slotz",
    "target_slotz",
    "attentd",
    "dynamic_attentd",
]
ckpt_map = [
    ["m.discov.", "m."],
]
freez = [r"^m\.discov\..*"]

param_groups = None
optimiz = dict(type=Adam, params=param_groups, lr=lr)
gscale = dict(type=GradScaler)
gclip = None

loss_fn_t = loss_fn_v = dict(
    dynamic=dict(
        metric=dict(type=DynamicsLoss),
        map=dict(input="output.slotz", target="output.target_slotz"),
    ),
)

def _flatten_segment_bt_hw(x):
    if x.ndim == 5:
        return rearrange(x, "b t h w s -> b (t h w) s")
    if x.ndim == 4:
        return rearrange(x, "t h w s -> 1 (t h w) s")
    raise ValueError(f"segment tensor must be 4D or 5D, got shape={tuple(x.shape)}")

def _flatten_segment_keep_t(x):
    if x.ndim == 5:
        return rearrange(x, "b t h w s -> b t (h w) s")
    if x.ndim == 4:
        return rearrange(x, "t h w s -> 1 t (h w) s")
    raise ValueError(f"segment tensor must be 4D or 5D, got shape={tuple(x.shape)}")

def _flatten_input_and_target(*x):
    if len(x) == 1 and isinstance(x[0], (tuple, list)):
        x = tuple(x[0])
    if len(x) == 1:
        return _flatten_segment_bt_hw(x[0])
    return tuple(_flatten_segment_bt_hw(v) for v in x)

def _flatten_pred_and_target(*x):
    if len(x) == 1 and isinstance(x[0], (tuple, list)):
        x = tuple(x[0])
    if len(x) != 2:
        raise ValueError(f"expect (pred, target), got {len(x)} inputs")

    pred, target = x
    pred_t = pred.shape[1] if pred.ndim == 5 else pred.shape[0]

    if target.ndim == 5:
        target = target[:, -pred_t:, ...]
    elif target.ndim == 4:
        target = target[-pred_t:, ...]
    else:
        raise ValueError(f"target tensor must be 4D or 5D, got shape={tuple(target.shape)}")

    return _flatten_segment_bt_hw(pred), _flatten_segment_bt_hw(target)

_acc_dict_ = dict(

    map=dict(input="output.segment", target="batch.segment"),
    transform=dict(
        type=Lambda,
        ikeys=[["input"], ["target"]],
        okeys=[["input"], ["target"]],
        func=_flatten_input_and_target,
    ),
)
_acc_track_dict_ = dict(
    map=dict(input="output.segment", target="batch.segment"),
    transform=dict(
        type=Lambda,
        ikeys=[["input"], ["target"]],
        okeys=[["input"], ["target"]],
        func=lambda x, y: (_flatten_segment_keep_t(x), _flatten_segment_keep_t(y)),
    ),
)

_acc_pred_dict_ = dict(
    map=dict(input="output.dynamic_segment", target="batch.segment"),
    transform=dict(
        type=Lambda,
        ikeys=[["input"], ["target"]],
        okeys=[["input"], ["target"]],
        func=_flatten_pred_and_target,
    ),
)
acc_fn_t = dict(
    mbo=dict(metric=dict(type=mBO, skip=[]), **_acc_dict_),
)
acc_fn_v = dict(
    ari=dict(metric=dict(type=ARI, skip=[]), **_acc_dict_),
    ari_fg=dict(metric=dict(type=ARI, skip=[0]), **_acc_dict_),
    mbo=dict(metric=dict(type=mBO, skip=[]), **_acc_dict_),
    miou=dict(metric=dict(type=mIoU, skip=[]), **_acc_dict_),
    idf1=dict(
        metric=dict(type=IDF1TrackEval, threshold=0.5, field="IDF1", skip=[]),
        **_acc_track_dict_,
    ),
    hota=dict(
        metric=dict(type=HOTATrackEval, field="HOTA", skip=[]),
        **_acc_track_dict_,
    ),
    pred_ari=dict(metric=dict(type=ARI, skip=[]), **_acc_pred_dict_),
    pred_ari_fg=dict(metric=dict(type=ARI, skip=[0]), **_acc_pred_dict_),
    pred_mbo=dict(metric=dict(type=mBO, skip=[]), **_acc_pred_dict_),
    pred_miou=dict(metric=dict(type=mIoU, skip=[]), **_acc_pred_dict_),
)

before_step = [
    dict(
        type=Lambda,
        ikeys=[["batch.video", "batch.segment"]],
        func=lambda _: _.cuda(),
    ),
    dict(
        type=CbLinearCosine,
        assigns=["optimiz.param_groups[0]['lr']=value"],
        nlin=total_step // 20,
        ntotal=total_step,
        vstart=0,
        vbase=lr,
        vfinal=lr / 1e3,
    ),
]
after_forward = [
    dict(
        type=Lambda,
        ikeys=[["output.attentd"]],
        func=lambda _: ptnf.one_hot(
            interpolat_argmax_attent(_.detach(), size=resolut0).long()
        ).bool(),
        okeys=[["output.segment"]],
    ),
]

after_forward_v = [
    *after_forward,
    dict(
        type=Lambda,
        ikeys=[["output.dynamic_attentd"]],
        func=lambda _: ptnf.one_hot(
            interpolat_argmax_attent(_.detach(), size=resolut0).long()
        ).bool(),
        okeys=[["output.dynamic_segment"]],
    ),
]
callback_t = [
    dict(type=Callback, before_step=before_step, after_forward=after_forward),
    dict(type=AverageLog, log_file=...),
]
callback_v = [
    dict(type=Callback, before_step=before_step[:1], after_forward=after_forward_v),
    callback_t[1],
    dict(type=SaveModel, save_dir=..., since_step=total_step * 0.5),
]
