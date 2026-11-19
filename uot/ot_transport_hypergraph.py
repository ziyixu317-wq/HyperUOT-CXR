"""Minimal, zero-parameter OT incidence change for the official HyperACE detector.

Only AdaHyperedgeGen.forward changes. Its official soft incidence A0 defines
the transport prior P0=A0/K, rather than assigning meaning to arbitrary logit
column offsets. HyperACE, AdaHGComputation, AdaHGConv,
their parameters and both native message-passing formulas remain untouched.
The logits calculation is derived from iMoonLab/yolov13 (AGPL-3.0).
"""
from __future__ import annotations
import copy
import math
import torch
from ultralytics.nn.modules.block import AdaHyperedgeGen, HyperACE

BASE_CONFIG = {"epsilon": 1.0, "source_kl": None, "target_kl": 0.5, "sinkhorn_iters": 32}
VARIANT_CONFIGS = {
    "transport_soft_capacity": {"target_kl": 0.5},
    "transport_soft_source_rho05": {"source_kl": 0.5, "target_kl": 0.5},
    "transport_soft_source_rho2": {"source_kl": 2.0, "target_kl": 0.5},
    "transport_soft_source_rho01": {"source_kl": 0.1, "target_kl": 0.5},
    "transport_soft_source_rho01_tau01": {"source_kl": 0.1, "target_kl": 0.1},
    "transport_soft_source_rho05_tau01": {"source_kl": 0.5, "target_kl": 0.1},
    "transport_soft_source_rho2_tau01": {"source_kl": 2.0, "target_kl": 0.1},
    "transport_soft_source_rho01_tau2": {"source_kl": 0.1, "target_kl": 2.0},
    "transport_soft_source_rho05_tau2": {"source_kl": 0.5, "target_kl": 2.0},
    "transport_soft_source_rho2_tau2": {"source_kl": 2.0, "target_kl": 2.0},
}
TRANSPORT_CLASS_NAMES = ("TransportHyperedgeGen",)


def variant_config(variant: str, config: dict | None = None) -> dict:
    if variant not in VARIANT_CONFIGS:
        raise ValueError(f"Unknown transport variant {variant!r}")
    unknown = set(config or {}) - set(BASE_CONFIG)
    if unknown:
        raise ValueError(f"Unknown transport config fields: {sorted(unknown)}")
    result = {**BASE_CONFIG, **VARIANT_CONFIGS[variant], **(config or {})}
    if not math.isfinite(result["epsilon"]) or result["epsilon"] <= 0:
        raise ValueError("epsilon must be finite and positive")
    if not math.isfinite(result["target_kl"]) or result["target_kl"] < 0:
        raise ValueError("target_kl must be finite and nonnegative")
    if result["source_kl"] is not None and (not math.isfinite(result["source_kl"]) or result["source_kl"] < 0):
        raise ValueError("source_kl must be None (hard rows) or finite and nonnegative")
    if not isinstance(result["sinkhorn_iters"], int) or result["sinkhorn_iters"] < 1:
        raise ValueError("sinkhorn_iters must be a positive integer")
    return result


def hard_source_soft_target(scores: torch.Tensor, source: torch.Tensor,
                            target_prior: torch.Tensor, epsilon: float = 1.0,
                            target_kl: float = 0.5, iterations: int = 32) -> torch.Tensor:
    """Solve min -<S,P>+eps sum P(log P-1)+tau KL(P^T1||b), P1=a.

    Source rows are hard constraints. Target columns are ONLY a KL prior.
    In particular, never renormalize P columns to b. At tau=0 this has the
    closed form P=a*softmax(S/eps, dim=edges), a row-softmax control rather
    than a nontrivial globally coupled OT solver.
    """
    if scores.ndim != 3 or source.shape != scores.shape[:2] or target_prior.shape != (scores.shape[0], scores.shape[2]):
        raise ValueError("Expected scores[B,N,K], source[B,N], target_prior[B,K]")
    if epsilon <= 0 or target_kl < 0 or iterations < 1:
        raise ValueError("Invalid transport solver configuration")
    with torch.autocast(device_type=scores.device.type, enabled=False):
        s, a, b = scores.float(), source.float(), target_prior.float()
        log_a = a.clamp_min(1e-30).log()
        if target_kl == 0:
            log_p = log_a[:, :, None] + torch.log_softmax(s / epsilon, dim=2)
        else:
            log_k = (s - s.amax(dim=(1, 2), keepdim=True)) / epsilon
            log_b = b.clamp_min(1e-30).log()
            log_v = torch.zeros_like(log_b)
            relaxation = target_kl / (target_kl + epsilon)
            for _ in range(iterations):
                log_u = log_a - torch.logsumexp(log_k + log_v[:, None, :], dim=2)
                log_v = relaxation * (log_b - torch.logsumexp(log_k + log_u[:, :, None], dim=1))
            # Finish with the HARD SOURCE update, not a target projection.
            log_u = log_a - torch.logsumexp(log_k + log_v[:, None, :], dim=2)
            log_p = log_k + log_u[:, :, None] + log_v[:, None, :]
        p = log_p.clamp_min(-80.0).exp()
        # Tiny float32 underflow protection and roundoff correction; only rows
        # are corrected, preserving the intended soft target capacity.
        return p * (a[:, :, None] / p.sum(dim=2, keepdim=True).clamp_min(1e-30))


def soft_source_soft_target(log_prior: torch.Tensor, source: torch.Tensor,
                            target_prior: torch.Tensor, source_kl: float = .5,
                            target_kl: float = .5, epsilon: float = 1.,
                            iterations: int = 32) -> torch.Tensor:
    """Solve eps KL(P||P0)+rho KL(P1||a)+tau KL(P^T1||b).

    log_prior is log(P0), including its meaningful global mass scale.
    There is NO global-max shift and NO final row or column projection.
    """
    if log_prior.ndim != 3 or source.shape != log_prior.shape[:2] or target_prior.shape != (log_prior.shape[0],log_prior.shape[2]):
        raise ValueError("Expected log_prior[B,N,K], source[B,N], target_prior[B,K]")
    if epsilon <= 0 or source_kl < 0 or target_kl < 0 or iterations < 1:
        raise ValueError("Invalid dual-soft transport configuration")
    with torch.autocast(device_type=log_prior.device.type,enabled=False):
        log_k=log_prior.float()
        log_a=source.float().clamp_min(1e-30).log()
        log_b=target_prior.float().clamp_min(1e-30).log()
        log_v=torch.zeros_like(log_b)
        source_factor=source_kl/(source_kl+epsilon)
        target_factor=target_kl/(target_kl+epsilon)
        for _ in range(iterations):
            log_u=source_factor*(log_a-torch.logsumexp(log_k+log_v[:,None,:],dim=2))
            log_v=target_factor*(log_b-torch.logsumexp(log_k+log_u[:,:,None],dim=1))
        return (log_k+log_u[:,:,None]+log_v[:,None,:]).clamp_min(-80.).exp()


class TransportHyperedgeGen(AdaHyperedgeGen):
    """Official learned generator, with its incidence normalization replaced."""
    def __init__(self, node_dim, num_hyperedges, num_heads=4, dropout=0.1,
                 context="both", variant="transport_soft_capacity", config=None):
        super().__init__(node_dim, num_hyperedges, num_heads, dropout, context)
        self.configure_transport(variant, config)

    def configure_transport(self, variant, config=None):
        self.transport_variant = variant
        self.transport_config = variant_config(variant, config)
        self.capture_transport = False
        self.clear_transport_cache()

    @classmethod
    def from_original(cls, original, variant, config=None):
        result = cls.__new__(cls)
        result.__dict__ = copy.deepcopy(original.__dict__)
        result.configure_transport(variant, config)
        return result

    def clear_transport_cache(self):
        for name in ("cached_transport", "cached_source", "cached_target_prior",
                     "cached_actual_column_mass", "cached_assignment", "cached_original_assignment",
                     "cached_prior_transport",
                     "cached_epsilon", "cached_target_kl", "cached_source_error"):
            setattr(self, name, None)
        self.cached_source_kl=None
        self.cached_actual_row_mass=None
        self.cached_total_mass=None

    def official_logits(self, X):
        # Preserve the official context/prototype/projection/scaling/dropout
        # operations and their original parameter names exactly.
        B, N, D = X.shape
        if self.context == "mean":
            context_cat = X.mean(dim=1)
        elif self.context == "max":
            context_cat, _ = X.max(dim=1)
        else:
            avg_context = X.mean(dim=1)
            max_context, _ = X.max(dim=1)
            context_cat = torch.cat([avg_context, max_context], dim=-1)
        prototype_offsets = self.context_net(context_cat).view(B, self.num_hyperedges, D)
        prototypes = self.prototype_base.unsqueeze(0) + prototype_offsets
        X_proj = self.pre_head_proj(X)
        X_heads = X_proj.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        proto_heads = prototypes.view(B, self.num_hyperedges, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        X_heads_flat = X_heads.reshape(B * self.num_heads, N, self.head_dim)
        proto_heads_flat = proto_heads.reshape(B * self.num_heads, self.num_hyperedges, self.head_dim).transpose(1, 2)
        logits = torch.bmm(X_heads_flat, proto_heads_flat) / self.scaling
        logits = logits.view(B, self.num_heads, N, self.num_hyperedges).mean(dim=1)
        return self.dropout(logits)

    def forward(self, X):
        logits = self.official_logits(X)
        B, N, _ = logits.shape
        cfg = self.transport_config
        with torch.autocast(device_type=X.device.type, enabled=False):
            source = torch.full((B, N), 1.0 / N, device=X.device, dtype=torch.float32)
            prior = torch.full((B, self.num_hyperedges), 1.0 / self.num_hyperedges,
                               device=X.device, dtype=torch.float32)
            # With epsilon=1 this is KL(P || P0)+tau KL(P^T1 || b),
            # P0=A0/K. The constant -log(K) is immaterial under P1=a.
            # log_softmax also removes the column-offset gauge that official
            # node-axis softmax cannot identify in the first place.
            log_official_incidence = logits.float().log_softmax(dim=1)
            if cfg["source_kl"] is None:
                coupling = hard_source_soft_target(log_official_incidence, source, prior,
                    cfg["epsilon"], cfg["target_kl"], cfg["sinkhorn_iters"])
            else:
                coupling = soft_source_soft_target(log_official_incidence-math.log(self.num_hyperedges),
                    source,prior,cfg["source_kl"],cfg["target_kl"],cfg["epsilon"],cfg["sinkhorn_iters"])
            actual_columns = coupling.sum(dim=1)
            assignment = coupling / actual_columns[:, None, :].clamp_min(1e-30)
            if self.capture_transport:
                self.cached_transport = coupling.detach().cpu()
                self.cached_source = source.detach().cpu()
                self.cached_target_prior = prior.detach().cpu()
                self.cached_actual_column_mass = actual_columns.detach().cpu()
                self.cached_assignment = assignment.detach().cpu()
                self.cached_original_assignment = logits.float().softmax(dim=1).detach().cpu()
                self.cached_prior_transport = self.cached_original_assignment / self.num_hyperedges
                self.cached_epsilon = cfg["epsilon"]
                self.cached_target_kl = cfg["target_kl"]
                self.cached_source_kl = cfg["source_kl"]
                self.cached_actual_row_mass = coupling.sum(dim=2).detach().cpu()
                self.cached_total_mass = coupling.sum(dim=(1,2)).detach().cpu()
                self.cached_source_error = float((coupling.sum(dim=2) - source).abs().max().detach())
        return assignment.to(dtype=X.dtype)


def _target_model(model):
    return model.model if hasattr(model, "model") and hasattr(model.model, "model") else model


def apply_transport(model, variant, config=None):
    """After official 17-class get_model/load and before EMA, replace 2 generators.

    No new parameters, buffers, projections, hyperedges, gates or message passing
    layers are added. Only two explicitly serialized generator instances differ.
    """
    target = _target_model(model)
    neck = target.model[9]
    if type(neck) is not HyperACE or list(neck.f) != [4, 6, 8]:
        raise ValueError("Expected original HyperACE layer9 from [4,6,8]")
    names_before = set(dict(target.named_parameters()))
    all_before = {name: value.detach().clone() for name, value in target.state_dict().items()}
    for branch in (neck.branch1, neck.branch2):
        original = branch.m.hgnn.edge_generator
        if type(original) is not AdaHyperedgeGen:
            raise ValueError("Expected an unmodified original edge generator")
        branch.m.hgnn.edge_generator = TransportHyperedgeGen.from_original(original, variant, config)
    all_after = target.state_dict()
    assert set(all_before) == set(all_after)
    assert names_before == set(dict(target.named_parameters()))
    assert all(torch.equal(value, all_after[name]) for name, value in all_before.items())
    report = {"variant": variant, "config": variant_config(variant, config), "new_parameters": 0,
              "new_parameter_names": [], "new_state_keys": [], "generator_count": 2,
              "all_model_state_tensors_identical": True, "state_tensor_count": len(all_before),
              "all_parameter_names_identical": True, "neck_class_unchanged": type(neck).__name__,
              "custom_serialized_class": "ot_transport_hypergraph.TransportHyperedgeGen",
              "original_message_passing_unchanged": True,
              "cost_basis": "Official incidence KL prior P0=A0/K; no mass projection in dual-soft variants",
              "source_constraint": "hard" if variant_config(variant,config)["source_kl"] is None else "soft KL",
              "scope": "Only AdaHyperedgeGen incidence normalization; fused HyperACE features"}
    target.transport_variant = variant
    target.transport_conversion_report = report
    return report


def set_transport_capture(model, enabled=True):
    for module in model.modules():
        if isinstance(module, TransportHyperedgeGen):
            module.capture_transport = bool(enabled)
            module.clear_transport_cache()


def transport_generators(model):
    return {name: module for name, module in model.named_modules()
            if isinstance(module, TransportHyperedgeGen)}
