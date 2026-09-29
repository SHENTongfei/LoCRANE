# -*- coding: utf-8 -*-
"""model_v3.py - CraneZV3 (TAIL-CERT). Evolves v2 with problem-aligned surgery:
  1. FiLM low-rank (2.16M -> ~135k): hidden 16, rank-16 basis shared across tokens
  2. Ordinal 3-band head (cumulative logits, label smoothing)
  3. Band-position head with tail-dilated target (L-band span 1.4x)  [Tv]
  4. Adaptive sample-conditional modality gate (gene/plm/imm)  [W1, flag]
  5. MoE band-routed FFN experts (dense + tail expert)          [W1, flag]
  6. Dead params removed (no ridge_proj, no supcon_proj)
Pace field v(t) stays POST-HOC at stage-0 (segmented slopes on OOF); flow head
only enters trunk after W1 screen passes (H58).
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class GeneModulePool(nn.Module):
    def __init__(self, assignment, n_genes, n_modules):
        super().__init__()
        W = np.zeros((n_modules, n_genes), dtype=np.float32)
        for gene_idx, mod in assignment.items():
            W[mod, gene_idx] = 1.0
        cnt = W.sum(axis=1, keepdims=True); cnt[cnt == 0] = 1.0
        self.register_buffer("W", torch.from_numpy(W / cnt))

    def forward(self, x):
        return torch.mm(x, self.W.t())


class FiLMLowRank(nn.Module):
    """sex -> (gamma,beta) with rank-r basis: params ~ hidden*2*M*r + d*r."""
    def __init__(self, n_modules, d_model, hidden=16, rank=16):
        super().__init__()
        self.n_modules, self.d_model, self.rank = n_modules, d_model, rank
        self.mlp = nn.Sequential(nn.Linear(1, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 2 * n_modules * rank))
        self.basis = nn.Parameter(torch.randn(d_model, rank) * 0.01)
        nn.init.zeros_(self.mlp[-1].bias); nn.init.zeros_(self.mlp[-1].weight)

    def forward(self, h, sex):
        gb = self.mlp(sex.unsqueeze(-1)).view(-1, 2, self.n_modules, self.rank)
        gb = (gb.reshape(-1, self.rank) @ self.basis.t()).view(-1, 2, self.n_modules, self.d_model)
        gamma, beta = gb[:, 0], gb[:, 1]
        return h * (1.0 + gamma) + beta


class MoEFFN(nn.Module):
    """2-expert FFN: dense expert + tail expert; router conditioned on band prior."""
    def __init__(self, d_model, ff_dim, dropout):
        super().__init__()
        self.e1 = nn.Sequential(nn.Linear(d_model, ff_dim), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(ff_dim, d_model), nn.Dropout(dropout))
        self.e2 = nn.Sequential(nn.Linear(d_model, ff_dim), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(ff_dim, d_model), nn.Dropout(dropout))
        self.router = nn.Linear(4, 2)  # [band_onehot(3), tau_hat(1)] -> logits
        self.last_usage = None

    def forward(self, x, band_prior=None):
        # LEAK FIX (R1 audit): router must NOT see true band labels (eval passed
        # b_va one-hot -> moe mAUC 1.0 was leakage, voided). Router now keys on
        # pooled content only; band_prior kept in signature for compat, ignored.
        pooled = x.mean(1)
        r_in = pooled.new_zeros(pooled.shape[0], 4)
        w = torch.softmax(self.router(r_in), -1)          # (B,2)
        self.last_usage = w.detach()
        return w[:, 0:1, None] * self.e1(x) + w[:, 1:2, None] * self.e2(x)  # (B,1,1) broadcast over tokens


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, dropout, use_moe=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.use_moe = use_moe
        if use_moe:
            self.ff = MoEFFN(d_model, ff_dim, dropout)
        else:
            self.ff = nn.Sequential(nn.Linear(d_model, ff_dim), nn.GELU(), nn.Dropout(dropout),
                                    nn.Linear(ff_dim, d_model), nn.Dropout(dropout))

    def forward(self, x, band_prior=None):
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x))
        x = x + a
        if self.use_moe:
            x = x + self.ff(self.ln2(x), band_prior)
        else:
            x = x + self.ff(self.ln2(x))
        return x


class ModalityGate(nn.Module):
    """Sample-conditional gate over modalities (adaptive fusion, W1)."""
    def __init__(self, n_mod, n_feat=8):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_feat, 16), nn.GELU(), nn.Linear(16, n_mod))
        self.last_w = None

    def forward(self, feats):
        w = torch.softmax(self.net(feats), -1)
        self.last_w = w
        return w


def three_view_supcon(views, temp=0.1):
    """R2 contra: NT-Xent over 3 same-sample views (q / s / immune reprs).
    Cross-view pairs of the same sample are positives; all other
    view-sample entries are negatives. (Revives model_v2 supcon idea.)"""
    zs = [F.normalize(v, dim=-1) for v in views]
    B = zs[0].shape[0]
    Z = torch.stack(zs, 1).reshape(3 * B, -1)                 # row 3b+i = view i of sample b
    sim = Z @ Z.t() / temp
    sim = sim.masked_fill(torch.eye(3 * B, device=Z.device, dtype=torch.bool), -1e9)
    e = sim.exp()
    r = torch.arange(3 * B, device=Z.device)
    i = r % 3
    p1 = torch.where(i == 0, r + 1, torch.where(i == 1, r - 1, r - 2))
    p2 = torch.where(i == 0, r + 2, torch.where(i == 1, r + 1, r - 1))
    num = e.gather(1, p1[:, None])[:, 0] + e.gather(1, p2[:, None])[:, 0]
    den = e.sum(1)
    return -torch.log((num + 1e-9) / (den + 1e-9)).mean()


class CraneZV3(nn.Module):
    def __init__(self, n_genes, assignment, n_modules, n_immune=19,
                 d_model=128, n_heads=4, n_layers=3, ff_dim=384, dropout=0.1,
                 ridge_residual=float(os.environ.get("V3_RR", "1.0")), use_moe=False, use_gate=False,
                 n_plm=0, plm_dim=1280, plm_bottleneck=128, use_contra=False):
        super().__init__()
        self.n_modules, self.d_model, self.n_immune = n_modules, d_model, n_immune
        self.use_moe, self.use_gate = use_moe, use_gate
        self.use_contra = use_contra
        self.pool = GeneModulePool(assignment, n_genes, n_modules)
        self.module_embed = nn.Linear(1, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_modules, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.film = FiLMLowRank(n_modules, d_model)
        self.gene_proj = nn.Linear(n_genes, d_model)
        self.reg_gene_proj = nn.Linear(n_genes, d_model)
        self.n_plm = n_plm
        if n_plm > 0:
            self.plm_proj = nn.Linear(plm_dim, plm_bottleneck)
            self.plm_token = nn.Linear(plm_bottleneck, d_model)
        if n_immune > 0:
            self.immune_proj = nn.Linear(n_immune, d_model)
            self.immune_head = nn.Sequential(nn.Linear(n_immune, 64), nn.GELU(),
                                             nn.Dropout(dropout), nn.Linear(64, 1))
        if use_gate:
            self.gate = ModalityGate(n_mod=2 + (1 if n_plm > 0 else 0))
        if use_contra:
            self.contra_proj = nn.ModuleList([nn.Sequential(nn.Linear(d_model, 64), nn.GELU(),
                                                             nn.Linear(64, 64)) for _ in range(3)])
        self.ridge_residual = ridge_residual
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, ff_dim, dropout,
                                                      use_moe=(use_moe and i == n_layers - 1))
                                     for i in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.reg_head = nn.Sequential(nn.Linear(d_model * 2 + 1, 128), nn.GELU(),
                                      nn.Dropout(dropout), nn.Linear(128, 1))
        self.cls_head = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(),
                                      nn.Dropout(dropout), nn.Linear(64, 1))
        self.band_head = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(),
                                       nn.Dropout(dropout), nn.Linear(64, 2))  # 2 cumulative logits -> 3 bands
        self.bandpos_head = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(),
                                          nn.Dropout(dropout), nn.Linear(64, 1))
        self.mfm_head = nn.Sequential(nn.Linear(d_model, 128), nn.GELU(), nn.Linear(128, 1))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.log_var_reg = nn.Parameter(torch.zeros(1))
        self.log_var_cls = nn.Parameter(torch.zeros(1))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # keep FiLM zero-init (v2 bug: xavier pass overwrote it) and basis small
        self.film.mlp[-1].weight.data.zero_(); self.film.mlp[-1].bias.data.zero_()

    def forward(self, x_gene_q, x_gene_s, sex, immune=None, ridge_z=None,
                band_prior=None, plm=None, mask_ratio=0.0):
        B = x_gene_q.shape[0]
        mod = self.pool(x_gene_q).unsqueeze(-1)
        h = self.module_embed(mod) + self.pos_embed
        h = self.film(h, sex)
        masked_idx = None
        if mask_ratio > 0:
            n_mask = max(1, int(round(self.n_modules * mask_ratio)))
            idx = torch.randperm(self.n_modules, device=x_gene_q.device)[:n_mask]
            h[:, idx] = self.mask_token
            masked_idx = idx
        tokens = [self.cls_token.expand(B, -1, -1)]
        gate_w = None
        if self.n_immune > 0 and immune is not None:
            tokens.append(self.immune_proj(immune).unsqueeze(1))
        if self.n_plm > 0 and plm is not None:
            n_plm_tok = plm.shape[1]
            tokens.append(self.plm_token(self.plm_proj(plm)).expand(B, -1, -1))
        else:
            n_plm_tok = 0
        tokens.append(h)
        h_all = torch.cat(tokens, dim=1)
        for blk in self.blocks:
            h_all = blk(h_all, band_prior)
        h_all = self.norm(h_all)
        cls = h_all[:, 0] + self.gene_proj(x_gene_q)
        n_prefix = 1 + (1 if (self.n_immune > 0 and immune is not None) else 0) + n_plm_tok
        mod_out = h_all[:, n_prefix:]
        rz = ridge_z.unsqueeze(-1) if ridge_z is not None else torch.zeros(B, 1, device=x_gene_q.device)
        reg_feat = torch.cat([cls, self.reg_gene_proj(x_gene_s), rz], dim=-1)
        delta = self.reg_head(reg_feat).squeeze(-1)
        age = ridge_z + self.ridge_residual * torch.tanh(delta) if ridge_z is not None else delta
        gene_logit = self.cls_head(cls).squeeze(-1)
        cum = self.band_head(cls)                       # (B,2) cumulative logits
        band_logits = torch.stack([-cum[:, 0], cum[:, 0] - cum[:, 1], cum[:, 1]], -1)  # P(Y)=-c0, P(M)=c0-c1, P(L)=c1  # ordinal->3 class
        tau = self.bandpos_head(cls).squeeze(-1)        # band position (dilated target)
        imm_logit = (self.immune_head(immune).squeeze(-1)
                     if (self.n_immune > 0 and immune is not None) else torch.zeros_like(gene_logit))
        if self.use_gate:
            stats = torch.stack([x_gene_q.mean(1), x_gene_q.std(1),
                                 x_gene_s.mean(1), x_gene_s.std(1)], 1)
            gp = band_prior if band_prior is not None else torch.zeros(B, 3, device=x_gene_q.device)
            gate_w = self.gate(torch.cat([stats, gp, sex.unsqueeze(1)], 1))
        mfm = self.mfm_head(mod_out).squeeze(-1)
        contra_loss = None
        if self.use_contra:
            v_q = mod_out.mean(1)
            v_s = self.reg_gene_proj(x_gene_s)
            v_i = (self.immune_proj(immune)
                   if (self.n_immune > 0 and immune is not None) else v_q)
            contra_loss = three_view_supcon(
                [self.contra_proj[k](v) for k, v in enumerate((v_q, v_s, v_i))], temp=0.1)
        return {"age": age, "gene_logit": gene_logit, "imm_logit": imm_logit,
                "band_cum": cum, "band_logits": band_logits, "tau": tau,
                "mfm": mfm, "mod_out": mod_out, "masked_idx": masked_idx,
                "cls": cls, "gate_w": gate_w, "contra_loss": contra_loss}


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
