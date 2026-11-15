from einops import rearrange, repeat
import numpy as np
import torch as pt
import torch.nn as nn

from ..util_learn import hungarian_matching, intersection_over_union

class ObjDynamicPredictor(nn.Module):

    def __init__(
        self,
        discov: nn.Module,
        dynamic: nn.Module,
        feature_idx: int = 0,
        slotz_idx: int = 1,
        attentd_idx: int = 4,
    ):
        super().__init__()
        self.discov = discov
        self.dynamic = dynamic
        self._decode_mode = None
        self._out_idx = {
            "feature": feature_idx,
            "slotz": slotz_idx,
            "attentd": attentd_idx,
        }

    def _unpack_discov_output(self, output):
        return (
            output[self._out_idx["feature"]],
            output[self._out_idx["slotz"]],
            output[self._out_idx["attentd"]],
        )

    def _decode_slots(self, slots: pt.Tensor, feature: pt.Tensor):

        b, t, n, c = slots.shape
        _, tf, _, h, w = feature.shape

        slots_flat = slots.flatten(0, 1)
        clue = rearrange(feature[:, -t:], "b t c h w -> (b t) (h w) c")
        recon, attentd = self._decode_with_discov(slots_flat, clue, h, w)

        recon = rearrange(recon, "(b t) (h w) c -> b t c h w", b=b, h=h)
        attentd = rearrange(attentd, "(b t) n (h w) -> b t n h w", b=b, h=h)
        return recon, attentd

    @pt._dynamo.disable
    def _decode_with_discov(self, slots_flat: pt.Tensor, clue: pt.Tensor, h: int, w: int):

        if self._decode_mode == "hw":
            return self.discov.decode([h, w], slots_flat)
        if self._decode_mode == "clue":
            return self.discov.decode(clue, slots_flat)

        errors = []
        for mode in ["hw", "clue"]:
            decode_input = [h, w] if mode == "hw" else clue
            try:
                recon, attentd = self.discov.decode(decode_input, slots_flat)
                self._decode_mode = mode
                return recon, attentd
            except Exception as exc:
                errors.append(f"{mode}: {exc}")

    def forward(self, input, condit=None):

        with pt.no_grad():
            discov_output = self.discov(input, condit)
            feature, slotz_gt, attentd = self._unpack_discov_output(discov_output)

        dynamic_output = self.dynamic(slotz_gt)
        slotz_pred = dynamic_output["next_state"]

        if self.training:
            dynamic_attentd = None
        else:
            _, dynamic_attentd = self._decode_slots(slotz_pred, feature)

        return slotz_pred, slotz_gt, attentd, dynamic_attentd

class ObjDynamicPredictorMerge2Slot(nn.Module):

    def __init__(
        self,
        discov: nn.Module,
        dynamic: nn.Module,
        feature_idx: int = 0,
        slotz_idx: int = 1,
        local_slotz: int = 2,
        attentd_idx: int = 4,
        slot_dim: int = 64,
    ):
        super().__init__()
        self.discov = discov
        self.dynamic = dynamic
        self._decode_mode = None
        self._out_idx = {
            "feature": feature_idx,
            "slotz": slotz_idx,
            "local_slotz": local_slotz,
            "attentd": attentd_idx,
        }
        self.slot_merge = nn.Sequential(
            nn.Linear(slot_dim * 2, slot_dim),
            nn.LayerNorm(slot_dim),
        )
    def _unpack_discov_output(self, output):
        return (
            output[self._out_idx["feature"]],
            output[self._out_idx["slotz"]],
            output[self._out_idx["local_slotz"]],
            output[self._out_idx["attentd"]],
        )

    def _decode_slots(self, slots: pt.Tensor, feature: pt.Tensor):

        b, t, n, c = slots.shape
        _, tf, _, h, w = feature.shape

        slots_flat = slots.flatten(0, 1)
        clue = rearrange(feature[:, -t:], "b t c h w -> (b t) (h w) c")
        recon, attentd = self._decode_with_discov(slots_flat, clue, h, w)

        recon = rearrange(recon, "(b t) (h w) c -> b t c h w", b=b, h=h)
        attentd = rearrange(attentd, "(b t) n (h w) -> b t n h w", b=b, h=h)
        return recon, attentd

    @pt._dynamo.disable
    def _decode_with_discov(self, slots_flat: pt.Tensor, clue: pt.Tensor, h: int, w: int):

        if self._decode_mode == "hw":
            return self.discov.decode([h, w], slots_flat)
        if self._decode_mode == "clue":
            return self.discov.decode(clue, slots_flat)

        errors = []
        for mode in ["hw", "clue"]:
            decode_input = [h, w] if mode == "hw" else clue
            try:
                recon, attentd = self.discov.decode(decode_input, slots_flat)
                self._decode_mode = mode
                return recon, attentd
            except Exception as exc:
                errors.append(f"{mode}: {exc}")

    def forward(self, input, condit=None):

        with pt.no_grad():
            discov_output = self.discov(input, condit)
            feature, slotz_id, local_slotz, attentd = self._unpack_discov_output(discov_output)
        slotz_gt = self.slot_merge(
            pt.cat([slotz_id, local_slotz], dim=-1)
        )
        dynamic_output = self.dynamic(slotz_gt)
        slotz_pred = dynamic_output["next_state"]

        if self.training:
            dynamic_attentd = None
        else:
            _, dynamic_attentd = self._decode_slots(slotz_pred, feature)

        return slotz_pred, slotz_gt, attentd, dynamic_attentd
