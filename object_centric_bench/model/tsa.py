import math

from einops import rearrange
import torch as pt
import torch.nn as nn
import torch.nn.functional as ptnf

from .dias import ARRandTransformerDecoder

class SlotGateMLPTSA(nn.Module):

    def __init__(self, in_dim, ctx_dim, hidden_dim=None, init_bias=2.0):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(1, in_dim // 2)
        self.in_dim = in_dim
        self.ctx_dim = ctx_dim
        self.fc1 = nn.Linear(in_dim + ctx_dim, hidden_dim)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, 1)
        with pt.no_grad():
            if self.fc2.bias is not None:
                self.fc2.bias.fill_(float(init_bias))

    def forward(self, slot_candidate, context, return_log=False):

        x = pt.cat([slot_candidate, context], dim=-1)
        h = self.act(self.fc1(x))
        logit = self.fc2(h).squeeze(-1)
        alpha = logit.sigmoid()
        if return_log:
            log_alpha = ptnf.logsigmoid(logit)
            return alpha, log_alpha
        return alpha

class SlotGateContextGRUTSA(nn.Module):

    def __init__(self, slot_dim, hidden_dim=64):
        super().__init__()
        self.cell = nn.GRUCell(slot_dim, hidden_dim)
        self.slot_dim = slot_dim
        self.hidden_dim = hidden_dim

    def init_hidden(self, batch_size, num_slots, device, dtype):
        return pt.zeros(
            batch_size, num_slots, self.hidden_dim, device=device, dtype=dtype
        )

    def forward(self, slot_state, alpha_prev=None, delta=None, h_prev=None):
        b, n, c = slot_state.shape
        x = slot_state.reshape(b * n, c)
        h_flat = h_prev.reshape(b * n, self.hidden_dim)
        h_new = self.cell(x, h_flat)
        return h_new.reshape(b, n, self.hidden_dim)

class GateAwareARRandTransformerDecoder(ARRandTransformerDecoder):

    LOG_ALPHA_FLOOR = -20.0

    def _num_heads(self):
        return self.backbone.layers[0].multihead_attn.num_heads

    def _build_memory_mask(self, gate=None, log_alpha=None, m=None):
        if log_alpha is None:
            if gate is None:
                return None
            log_alpha = gate.clamp_min(1e-9).log()
        log_alpha = log_alpha.clamp(min=self.LOG_ALPHA_FLOOR, max=0.0)
        b, n = log_alpha.shape
        nhead = self._num_heads()
        mask = (
            log_alpha[:, None, None, :]
            .expand(b, nhead, m, n)
            .reshape(b * nhead, m, n)
        )
        return mask.contiguous()

    def forward(self, input, slots, gate=None, log_alpha=None, smask=None, p=0.5):
        b, m, c = input.shape
        assert m == self.posit_embed.pe.size(1)
        _, n, _ = slots.shape
        device = input.device
        tokens = self.project1(input)

        if self.training:
            idxs = pt.vmap(
                lambda _: pt.randperm(m, device=device), randomness="different"
            )(tokens)
            idxs_expanded = idxs[:, :, None].expand(-1, -1, c)

            idxs0 = pt.arange(0, m, device=device)[None, :]
            keep1 = pt.randint(0, m - 1, [b, 1], device=device)
            keep2 = pt.ones(b, 1, dtype=pt.long, device=device) * int(256 * 0.1) - 1
            cond = pt.rand(b, 1, device=device) < p
            keep = pt.where(cond, keep1, keep2)
            mask_keep = idxs0 < keep

            tokens_shuffled = tokens.gather(1, idxs_expanded)
            mask_token_expanded = self.mask_token.expand(b, m, -1)
            tokens_masked = tokens_shuffled.where(
                mask_keep[:, :, None], mask_token_expanded
            )

            pe_expanded = self.posit_embed.pe[:, :m, :].expand(b, -1, -1)
            pe_shuffled = pe_expanded.gather(1, idxs_expanded)

            query = tokens_masked + pe_shuffled
        else:
            query = tokens + self.posit_embed.pe[:, :m, :]

        memory = self.project2(slots)

        memory_mask = self._build_memory_mask(
            gate=gate, log_alpha=log_alpha, m=m
        )

        autoreg = self.backbone(
            self.norm0(query),
            memory=memory,
            memory_mask=memory_mask,
            memory_key_padding_mask=None if smask is None else ~smask,
        )
        recon = self.readout(autoreg)
        _, _, d = recon.shape

        if self.training:
            idxs_inverse = idxs.argsort(1)[:, :, None]
            recon = recon.gather(1, idxs_inverse.expand(-1, -1, d))
            attent = self._attent.gather(1, idxs_inverse.expand(-1, -1, n))
        else:
            attent = self._attent

        attent = rearrange(attent, "b m n -> b n m")
        return recon, attent

class TSA(nn.Module):

    def __init__(
        self,
        encode_backbone,
        encode_posit_embed,
        encode_project,
        initializ,
        aggregat,
        transit,
        decode,
        gate,
        gate_context,
        state_interpolation=True,
        decoder_gating=True,
        alpha_prev_init=1.0,
    ):
        super().__init__()
        self.encode_backbone = encode_backbone
        self.encode_posit_embed = encode_posit_embed
        self.encode_project = encode_project
        self.initializ = initializ
        self.aggregat = aggregat
        self.transit = transit
        self.decode = decode
        self.gate = gate
        self.gate_context = gate_context

        self.state_interpolation = state_interpolation
        self.decoder_gating = decoder_gating

        self._decoder_is_gate_aware = isinstance(
            decode, GateAwareARRandTransformerDecoder
        )
        if not self._decoder_is_gate_aware:
            self.decoder_gating = False

        assert 0.0 < alpha_prev_init < 1.0 or alpha_prev_init in (0.0, 1.0), (
            f"alpha_prev_init must be in (0, 1), got {alpha_prev_init}"
        )
        api = min(max(alpha_prev_init, 1e-4), 1.0 - 1e-4)
        raw_init = math.log(api / (1.0 - api))
        self.alpha_prev_init_raw = nn.Parameter(pt.tensor([raw_init]))

        self.reset_parameters(
            [
                self.encode_posit_embed,
                self.encode_project,
                self.aggregat,
                self.transit,
            ]
        )
        for m in self.gate_context.modules():
            if isinstance(m, nn.GRUCell):
                nn.init.zeros_(m.bias_ih)
                nn.init.zeros_(m.bias_hh)

    @staticmethod
    def reset_parameters(modules):
        for module in modules:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.GRUCell):
                    if m.bias:
                        nn.init.zeros_(m.bias_ih)
                        nn.init.zeros_(m.bias_hh)

    def forward(self, input, condit=None):

        b, t, _, _, _ = input.shape
        input = input.flatten(0, 1)

        feature = self.encode_backbone(input).detach()
        _, _, h, _ = feature.shape

        encode = feature.permute(0, 2, 3, 1)
        encode = self.encode_posit_embed(encode)
        encode = encode.flatten(1, 2)
        encode = self.encode_project(encode)

        feature = rearrange(feature, "(b t) c h w -> b t c h w", b=b)
        encode = rearrange(encode, "(b t) hw c -> b t hw c", b=b)

        slotz_list = []
        gatez_list = []
        log_alpha_list = []
        attenta_list = []

        h_prev = None
        alpha_prev = None

        for i in range(t):
            if i == 0:
                query_i = self.initializ(
                    b if condit is None else condit[:, 0, :, :]
                )
            else:
                slotz_so_far = pt.stack(slotz_list, dim=1)
                query_i = self.transit(slotz_so_far, encode[:, : i + 1, :, :])

            niter = None if i == 0 else 1
            slotz_candidate, attenta_i = self.aggregat(
                encode[:, i, :, :], query_i, num_iter=niter
            )

            bsz, n_slot, _ = slotz_candidate.shape

            if h_prev is None:
                h_prev = self.gate_context.init_hidden(
                    bsz, n_slot, slotz_candidate.device, slotz_candidate.dtype
                )
                alpha_prev = (
                    self.alpha_prev_init_raw.sigmoid()
                    .to(dtype=slotz_candidate.dtype, device=slotz_candidate.device)
                    .expand(bsz, n_slot)
                    .contiguous()
                )

            alpha_i, log_alpha_i = self.gate(
                slotz_candidate, h_prev, return_log=True
            )

            if self.state_interpolation and i > 0:
                prev_slotz = slotz_list[-1]
                slotz_i = (
                    alpha_i[:, :, None] * slotz_candidate
                    + (1.0 - alpha_i[:, :, None]) * prev_slotz
                )
            else:
                slotz_i = slotz_candidate

            slotz_list.append(slotz_i)
            gatez_list.append(alpha_i)
            log_alpha_list.append(log_alpha_i)
            attenta_list.append(attenta_i)

            h_new = self.gate_context(
                slotz_i, alpha_prev=alpha_prev, delta=None, h_prev=h_prev
            )

            h_prev = h_new
            alpha_prev = alpha_i

        slotz = pt.stack(slotz_list, dim=1)
        gatez = pt.stack(gatez_list, dim=1)
        log_alpha = pt.stack(log_alpha_list, dim=1)
        attenta = pt.stack(attenta_list, dim=1)
        attenta = rearrange(attenta, "b t n (h w) -> b t n h w", h=h)

        clue = rearrange(feature, "b t c h w -> (b t) (h w) c")
        slots_flat = slotz.flatten(0, 1)
        if self.decoder_gating and self._decoder_is_gate_aware:
            log_alpha_flat = log_alpha.flatten(0, 1)
            recon, attentd = self.decode(clue, slots_flat, log_alpha=log_alpha_flat)
        else:
            recon, attentd = self.decode(clue, slots_flat)

        recon = rearrange(recon, "(b t) (h w) c -> b t c h w", b=b, h=h)
        attentd = rearrange(attentd, "(b t) n (h w) -> b t n h w", b=b, h=h)

        return feature, slotz, attenta, recon, attentd, gatez
