from einops import rearrange
import torch.nn.functional as ptnf

from object_centric_bench.datum import (
    StridedRandomSlice1,
    RandomCrop,
    Resize,
    RandomFlip,
    Normalize,
    CenterCrop,
    Lambda,
    OVIS,
    ClPadToMax1,
    DefaultCollate,
)
from object_centric_bench.learn import (
    Adam,
    GradScaler,
    ClipGradNorm,
    MSELoss,
    SlotContrastLoss,
    mBO,
    ARI,
    mIoU,
    HOTATrackEval,
    IDF1TrackEval,
    CbLinearCosine,
    Callback,
    AverageLog,
    SaveModel,
    GateBimodalLossTSA,
    GateUsageLossTSA,
)
from object_centric_bench.model import (
    Sequential,
    Interpolate,
    DINO2ViT,
    Identity,
    MLP,
    NormalShared,
    SlotAttention,
    RSFQTransit,
    GateAwareARRandTransformerDecoder,
    LearntPositionalEmbedding,
    Linear,
    LayerNorm,
    TransformerDecoder,
    TransformerDecoderLayer,
    TSA,
    SlotGateMLPTSA,
    SlotGateContextGRUTSA,
)
from object_centric_bench.util import Compose, ComposeNoStar
from object_centric_bench.util_model import interpolat_argmax_attent

max_num = 21 + 1
resolut0 = [256, 256]
resolut1 = [16, 16]
emb_dim = 256
vfm_dim = 384
gate_hidden = emb_dim // 2
ctx_dim = 64

total_step = 50000
val_interval = total_step // 40
batch_size_t = 32 // 4
batch_size_v = 1
num_work = 4
lr = 2e-4 / 4

lambda_ssc = 0.5
lambda_sparse_max = 0.15
lambda_usage_max = 0.15
lambda_warmup = 1000
lambda_ramp = 6000

def _lambda_by_step(step_count, vmax):
    if step_count < lambda_warmup:
        return 0.0
    if step_count >= lambda_warmup + lambda_ramp:
        return vmax
    return vmax * ((step_count - lambda_warmup) / lambda_ramp)

def _set_gate_loss_weights(loss_fn_t, loss_fn_v, step_count):
    w_sparse = _lambda_by_step(step_count, lambda_sparse_max)
    w_usage = _lambda_by_step(step_count, lambda_usage_max)
    loss_fn_t.metrics["sparse"]["weight"] = w_sparse
    loss_fn_v.metrics["sparse"]["weight"] = w_sparse
    loss_fn_t.metrics["usage"]["weight"] = w_usage
    loss_fn_v.metrics["usage"]["weight"] = w_usage
    return loss_fn_t, loss_fn_v

IMAGENET_MEAN = [[[123.675]], [[116.28]], [[103.53]]]
IMAGENET_STD = [[[58.395]], [[57.12]], [[57.375]]]
transform_t = [
    dict(type=StridedRandomSlice1, keys=["video", "segment"], dim=0, size=10),
    dict(type=RandomCrop, keys=["video", "segment"], size=None, scale=[0.75, 1]),
    dict(type=Resize, keys=["video"], size=resolut0, interp="bilinear"),
    dict(type=Resize, keys=["segment"], size=resolut0, interp="nearest-exact", c=0),
    dict(type=RandomFlip, keys=["video", "segment"], dims=[-1], p=0.5),
    dict(type=Normalize, keys=["video"], mean=[IMAGENET_MEAN], std=[IMAGENET_STD]),
]
transform_v = [
    dict(type=CenterCrop, keys=["video", "segment"], size=None),
    dict(type=Resize, keys=["video"], size=resolut0, interp="bilinear"),
    dict(type=Resize, keys=["segment"], size=resolut0, interp="nearest-exact", c=0),
    dict(type=Normalize, keys=["video"], mean=[IMAGENET_MEAN], std=[IMAGENET_STD]),
]
dataset_t = dict(
    type=OVIS,
    data_file="ovis/train.lmdb",
    ts=30,
    extra_keys=["segment"],
    transform=dict(type=Compose, transforms=transform_t),
    base_dir=...,
)
dataset_v = dict(
    type=OVIS,
    data_file="ovis/val.lmdb",
    ts=None,
    extra_keys=["segment"],
    transform=dict(type=Compose, transforms=transform_v),
    base_dir=...,
)
collate_fn_t = dict(
    type=ComposeNoStar,
    transforms=[
        dict(type=ClPadToMax1, keys=["segment"], dims=[3]),
        dict(type=DefaultCollate),
    ],
)
collate_fn_v = collate_fn_t

model = dict(
    type=TSA,
    encode_backbone=dict(
        type=Sequential,
        modules=[
            dict(type=Interpolate, scale_factor=0.875, interp="bicubic"),
            dict(
                type=DINO2ViT,
                model_name="vit_small_patch14_reg4_dinov2.lvd142m",
                in_size=int(resolut0[0] * 0.875),
                rearrange=True,
                norm_out=False,
            ),
        ],
    ),
    encode_posit_embed=dict(type=Identity),
    encode_project=dict(
        type=MLP, in_dim=vfm_dim, dims=[vfm_dim, vfm_dim], ln="pre", dropout=0.0
    ),
    initializ=dict(type=NormalShared, num=max_num, dim=emb_dim),
    aggregat=dict(
        type=SlotAttention,
        num_iter=3,
        embed_dim=emb_dim,
        ffn_dim=emb_dim * 4,
        dropout=0,
        kv_dim=vfm_dim,
        trunc_bp=None,
    ),
    transit=dict(
        type=RSFQTransit,
        dt=5,
        ci=vfm_dim,
        c=emb_dim,
        nhead=4,
        expanz=4,
        pdo=0.5,
        norm_first=False,
        bias=False,
    ),
    gate=dict(
        type=SlotGateMLPTSA,
        in_dim=emb_dim,
        ctx_dim=ctx_dim,
        hidden_dim=gate_hidden,
        init_bias=2.0,
    ),
    gate_context=dict(
        type=SlotGateContextGRUTSA,
        slot_dim=emb_dim,
        hidden_dim=ctx_dim,
    ),
    state_interpolation=True,
    decoder_gating=True,
    alpha_prev_init=1.0,
    decode=dict(
        type=GateAwareARRandTransformerDecoder,
        emb_dim=vfm_dim,
        posit_embed=dict(
            type=LearntPositionalEmbedding,
            resolut=[resolut1[0] * resolut1[1]],
            embed_dim=vfm_dim,
        ),
        project1=dict(
            type=Sequential,
            modules=[
                dict(
                    type=Linear, in_features=vfm_dim, out_features=vfm_dim, bias=False
                ),
                dict(type=LayerNorm, normalized_shape=vfm_dim),
            ],
        ),
        project2=dict(
            type=Sequential,
            modules=[
                dict(
                    type=Linear, in_features=emb_dim, out_features=vfm_dim, bias=False
                ),
                dict(type=LayerNorm, normalized_shape=vfm_dim),
            ],
        ),
        backbone=dict(
            type=TransformerDecoder,
            decoder_layer=dict(
                type=TransformerDecoderLayer,
                d_model=vfm_dim,
                nhead=4,
                dim_feedforward=vfm_dim * 4,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
                bias=False,
            ),
            num_layers=4,
        ),
        readout=dict(type=Identity),
    ),
)
model_imap = dict(input="batch.video")
model_omap = ["feature", "slotz", "attenta", "recon", "attentd", "gatez"]
ckpt_map = []
freez = [r"^m\.encode_backbone\..*"]

param_groups = None
optimiz = dict(type=Adam, params=param_groups, lr=lr)
gscale = dict(type=GradScaler)
gclip = dict(type=ClipGradNorm, max_norm=0.05)

loss_fn_t = loss_fn_v = dict(
    recon=dict(
        metric=dict(type=MSELoss),
        map=dict(input="output.recon", target="output.feature"),
        transform=dict(type=Lambda, ikeys=[["target"]], func=lambda _: _.detach()),
    ),
    ssc=dict(
        metric=dict(type=SlotContrastLoss),
        map=dict(input="output.slotz"),
        weight=lambda_ssc,
    ),
    sparse=dict(
        metric=dict(type=GateBimodalLossTSA),
        map=dict(input="output.gatez"),
        weight=0.0,
    ),
    usage=dict(
        metric=dict(type=GateUsageLossTSA),
        map=dict(input="output.gatez"),
        weight=0.0,
    ),
)
_acc_dict_ = dict(
    map=dict(input="output.segment", target="batch.segment"),
    transform=dict(
        type=Lambda,
        ikeys=[["input", "target"]],
        func=lambda _: rearrange(_, "b t h w s -> b (t h w) s"),
    ),
)
_acc_track_dict_ = dict(
    map=dict(input="output.segment", target="batch.segment"),
    transform=dict(
        type=Lambda,
        ikeys=[["input", "target"]],
        func=lambda _: rearrange(_, "b t h w s -> b t (h w) s"),
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
    dict(
        type=Lambda,
        ikeys=[["loss_fn_t"], ["loss_fn_v"], ["step_count"]],
        func=_set_gate_loss_weights,
        okeys=[["loss_fn_t"], ["loss_fn_v"]],
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
callback_t = [
    dict(type=Callback, before_step=before_step, after_forward=after_forward),
    dict(type=AverageLog, log_file=...),
]
callback_v = [
    dict(type=Callback, before_step=before_step[:1], after_forward=after_forward),
    callback_t[1],
    dict(type=SaveModel, save_dir=..., since_step=total_step * 0.5),
]
