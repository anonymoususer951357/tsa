from typing import Dict

import torch
from torch import nn

def build(config, name: str = None, default_name: str = None, **kwargs):
    if config is None:
        raise ValueError("No config specified while building dynamics_predictor")

    if name is None:
        name = config.get("name") if isinstance(config, dict) else None
    name = name or default_name
    if name is None:
        raise ValueError("No name specified while building dynamics_predictor")

    cls = globals().get(name)
    if cls is None or not issubclass(cls, DynamicsPredictor):
        raise ValueError(f"Unknown dynamics_predictor `{name}`")

    cfg = dict(config) if isinstance(config, dict) else {}
    cfg.pop("name", None)
    cfg.update(kwargs)
    return cls(**cfg)

class DynamicsPredictor(nn.Module):
    def __init__(self, history_len: int):
        super().__init__()
        self.history_len = history_len

    def forward(self, slots: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("This method should be implemented in the child class.")

class SlotFormer(DynamicsPredictor):

    def __init__(
        self,
        num_slots=7,
        slot_size=128,
        history_len=6,
        t_pe="sin",
        slots_pe="",
        d_model=128,
        num_layers=4,
        num_heads=8,
        ffn_dim=512,
        dropout=0.1,
        norm_first=True,
    ):
        super().__init__(history_len)

        self.num_slots = num_slots

        self.in_proj = nn.Linear(slot_size, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            norm_first=norm_first,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=enc_layer, num_layers=num_layers
        )

        self.enc_t_pe = self._build_pos_enc(t_pe, history_len, d_model)
        self.enc_slots_pe = self._build_pos_enc(slots_pe, num_slots, d_model)

        self.out_proj = nn.Linear(d_model, slot_size)

    def _build_pos_enc(self, pos_enc, input_len, d_model):

        if not pos_enc:
            return None
        if pos_enc == "learnable":
            pos_embedding = nn.Parameter(torch.zeros(1, input_len, d_model))
        elif "sin" in pos_enc:
            inv_freq = 1.0 / (10000 ** (torch.arange(0.0, d_model, 2.0) / d_model))
            pos_seq = torch.arange(input_len - 1, -1, -1).type_as(inv_freq)
            sinusoid_inp = torch.outer(pos_seq, inv_freq)
            pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
            pos_embedding = nn.Parameter(pos_emb.unsqueeze(0), requires_grad=False)
        else:
            raise NotImplementedError(f"{pos_enc} is not supported as a positional encoding.")
        return pos_embedding

    def forward(self, slots: torch.Tensor) -> Dict[str, torch.Tensor]:

        past_slots = slots[:, : self.history_len]
        rollout_len = slots.shape[1] - self.history_len

        past_slots = past_slots.flatten(1, 2)
        in_slots = past_slots
        enc_pe = (
            self.enc_t_pe.unsqueeze(2)
            .repeat(past_slots.shape[0], 1, self.num_slots, 1)
            .flatten(1, 2)
        )
        if self.enc_slots_pe is not None:
            slots_pe = (
                self.enc_slots_pe.unsqueeze(1)
                .repeat(past_slots.shape[0], self.history_len, 1, 1)
                .flatten(1, 2)
            )
            enc_pe = slots_pe + enc_pe

        pred_out = []
        for _ in range(rollout_len):
            past_slots = self.in_proj(in_slots)
            past_slots = past_slots + enc_pe
            past_slots = self.transformer_encoder(past_slots)
            pred_slots = self.out_proj(past_slots[:, -self.num_slots :])
            pred_out.append(pred_slots)
            in_slots = torch.cat([in_slots[:, self.num_slots :], pred_out[-1]], dim=1)

        pred_slots = torch.stack(pred_out, dim=1)
        return {"next_state": pred_slots}
