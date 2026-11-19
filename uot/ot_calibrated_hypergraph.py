"""Zero-parameter, calibrated OT incidence for the native HyperACE neck.

Only generator computation changes. Four existing HG projection weights have
a matched initialization repair; their biases, parameter paths/counts and the
official two-pass message-passing formulas remain unmodified.
Calibration is an explicit numerical/modeling change and has its own no-OT
control. Frozen ot_transport_hypergraph supplies the reviewed log-OT solvers.
"""
from __future__ import annotations
import copy
import hashlib
import math
import torch
import torch.nn.functional as F
from ultralytics.nn.modules.block import AdaHyperedgeGen, HyperACE
from ot_transport_hypergraph import (hard_source_soft_target, soft_source_soft_target,
                                    VARIANT_CONFIGS as LEGACY_VARIANTS)

BASE_CONFIG = {'epsilon': 1., 'source_kl': .5, 'target_kl': .5,
               'sinkhorn_iters': 32, 'calibration_temperature': 1.,
               'std_guard_ulps': 64., 'std_guard_absolute': 1e-12}
VARIANT_CONFIGS = {'calibrated_no_ot': {'source_kl': 0., 'target_kl': 0.}}
VARIANT_CONFIGS.update({key.replace('transport_', 'calibrated_', 1):
                       {'source_kl': None if key == 'transport_soft_capacity' else value.get('source_kl', .5), **value}
                       for key, value in LEGACY_VARIANTS.items()})
CALIBRATED_CLASS_NAMES = ('CalibratedTransportHyperedgeGen',)
RESET_WEIGHT_NAMES = tuple(f'model.9.{branch}.m.hgnn.{projection}.0.weight'
    for branch in ('branch1', 'branch2') for projection in ('edge_proj', 'node_proj'))
INITIALIZATION_SEED = 0


def variant_config(variant, config=None):
    if variant not in VARIANT_CONFIGS:
        raise ValueError(f'Unknown calibrated variant {variant!r}')
    if set(config or {}) - set(BASE_CONFIG):
        raise ValueError('Unknown calibrated configuration field')
    result = {**BASE_CONFIG, **VARIANT_CONFIGS[variant], **(config or {})}
    for name in ['epsilon', 'calibration_temperature', 'std_guard_ulps', 'std_guard_absolute']:
        if not math.isfinite(result[name]) or result[name] <= 0:
            raise ValueError(f'{name} must be finite and positive')
    for name in ['source_kl', 'target_kl']:
        value = result[name]
        if value is None and name == 'source_kl':
            continue
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    if not isinstance(result['sinkhorn_iters'], int) or result['sinkhorn_iters'] < 1:
        raise ValueError('sinkhorn_iters must be a positive integer')
    if variant == 'calibrated_no_ot' and (result['source_kl'] != 0 or result['target_kl'] != 0):
        raise ValueError('No-OT control must have both marginal penalties zero')
    return result


def calibrate_logits(logits, guard_ulps=64., absolute_guard=1e-12, temperature=1.):
    """Center and standardize each edge along nodes, with a rounding guard.

    An edge whose node std falls below the guard is explicitly degenerate and
    receives zero calibrated logits. It is not contrast-stretched from noise.
    std() has a finite zero-variance backward, unlike an unguarded sqrt(var).
    """
    logits = logits.float()
    mean = logits.mean(dim=1, keepdim=True)
    centered = logits - mean
    sigma = logits.std(dim=1, keepdim=True, unbiased=False)
    magnitude = logits.abs().amax(dim=1, keepdim=True)
    guard = (guard_ulps * torch.finfo(torch.float32).eps * magnitude).clamp_min(absolute_guard).detach()
    valid = sigma >= guard
    denominator = torch.maximum(sigma, guard)
    calibrated = torch.where(valid, centered / denominator / temperature, torch.zeros_like(centered))
    return calibrated, {'mean': mean, 'std': sigma, 'guard': guard,
                        'denominator': denominator, 'degenerate': ~valid}


class CalibratedTransportHyperedgeGen(AdaHyperedgeGen):
    """Original learned logits in FP32, calibrated before the original dropout."""
    def __init__(self, node_dim, num_hyperedges, num_heads=4, dropout=.1,
                 context='both', variant='calibrated_soft_source_rho05', config=None):
        super().__init__(node_dim, num_hyperedges, num_heads, dropout, context)
        self.configure_calibrated(variant, config)

    def configure_calibrated(self, variant, config=None):
        self.transport_variant = variant
        self.transport_config = variant_config(variant, config)
        self.capture_transport = False
        self.clear_transport_cache()

    @classmethod
    def from_original(cls, original, variant, config=None):
        result = cls.__new__(cls)
        result.__dict__ = copy.deepcopy(original.__dict__)
        result.configure_calibrated(variant, config)
        return result

    def clear_transport_cache(self):
        for name in ['cached_uncalibrated_assignment', 'cached_original_assignment',
                     'cached_calibrated_assignment', 'cached_prior_transport',
                     'cached_transport', 'cached_assignment', 'cached_source',
                     'cached_target_prior', 'cached_actual_column_mass',
                     'cached_actual_row_mass', 'cached_total_mass', 'cached_source_error',
                     'cached_raw_logits', 'cached_calibrated_logits', 'cached_dropout_logits',
                     'cached_calibration', 'cached_epsilon', 'cached_source_kl', 'cached_target_kl']:
            setattr(self, name, None)

    def raw_logits_fp32(self, X):
        """Same prototype/context/projection formula, no dropout, all FP32.

        Functional linear uses the original parameters at the original state
        paths; float casts also support a half-serialized eval checkpoint.
        """
        with torch.autocast(device_type=X.device.type, enabled=False):
            X = X.float()
            B, N, D = X.shape
            if self.context == 'mean':
                context_cat = X.mean(dim=1)
            elif self.context == 'max':
                context_cat = X.max(dim=1).values
            else:
                context_cat = torch.cat([X.mean(dim=1), X.max(dim=1).values], dim=-1)
            offsets = F.linear(context_cat, self.context_net.weight.float(),
                               None if self.context_net.bias is None else self.context_net.bias.float()).view(B, self.num_hyperedges, D)
            prototypes = self.prototype_base.float().unsqueeze(0) + offsets
            projected = F.linear(X, self.pre_head_proj.weight.float(),
                                 None if self.pre_head_proj.bias is None else self.pre_head_proj.bias.float())
            nodes = projected.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            edges = prototypes.view(B, self.num_hyperedges, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            logits = torch.bmm(nodes.reshape(B*self.num_heads, N, self.head_dim),
                               edges.reshape(B*self.num_heads, self.num_hyperedges, self.head_dim).transpose(1, 2)) / self.scaling
            return logits.view(B, self.num_heads, N, self.num_hyperedges).mean(dim=1)

    def forward(self, X):
        cfg = self.transport_config
        with torch.autocast(device_type=X.device.type, enabled=False):
            raw = self.raw_logits_fp32(X)
            normalized, calibration = calibrate_logits(raw, cfg['std_guard_ulps'],
                cfg['std_guard_absolute'], cfg['calibration_temperature'])
            # The inherited dropout is called ONCE, after centering/scaling.
            # In particular, constant column offsets cannot become fake node
            # evidence when dropout masks differ between nodes during training.
            scores = self.dropout(normalized)
            log_a0 = scores.log_softmax(dim=1)
            original = log_a0.exp()
            B, N, K = original.shape
            source = torch.full((B, N), 1/N, device=X.device, dtype=torch.float32)
            target = torch.full((B, K), 1/K, device=X.device, dtype=torch.float32)
            if self.transport_variant == 'calibrated_no_ot':
                coupling = original / K
                assignment = original
            elif cfg['source_kl'] is None:
                coupling = hard_source_soft_target(log_a0, source, target,
                    cfg['epsilon'], cfg['target_kl'], cfg['sinkhorn_iters'])
                assignment = coupling / coupling.sum(dim=1, keepdim=True).clamp_min(1e-30)
            else:
                coupling = soft_source_soft_target(log_a0-math.log(K), source, target,
                    cfg['source_kl'], cfg['target_kl'], cfg['epsilon'], cfg['sinkhorn_iters'])
                assignment = coupling / coupling.sum(dim=1, keepdim=True).clamp_min(1e-30)
            if self.capture_transport:
                def cpu(value):
                    return value.detach().cpu()
                self.cached_uncalibrated_assignment = cpu(raw.softmax(dim=1))
                self.cached_original_assignment = self.cached_uncalibrated_assignment
                self.cached_calibrated_assignment = cpu(original)
                self.cached_prior_transport = cpu(original / K)
                self.cached_transport = cpu(coupling)
                self.cached_assignment = cpu(assignment)
                self.cached_source = cpu(source)
                self.cached_target_prior = cpu(target)
                self.cached_actual_column_mass = cpu(coupling.sum(dim=1))
                self.cached_actual_row_mass = cpu(coupling.sum(dim=2))
                self.cached_total_mass = cpu(coupling.sum(dim=(1, 2)))
                self.cached_source_error = float((coupling.sum(dim=2)-source).abs().max().detach())
                self.cached_raw_logits = cpu(raw)
                self.cached_calibrated_logits = cpu(normalized)
                self.cached_dropout_logits = cpu(scores)
                self.cached_calibration = {key: cpu(value) for key, value in calibration.items()}
                self.cached_epsilon = cfg['epsilon']
                self.cached_source_kl = cfg['source_kl']
                self.cached_target_kl = cfg['target_kl']
        return assignment.to(X.dtype)


def _target_model(model):
    return model.model if hasattr(model, 'model') and hasattr(model.model, 'model') else model


def apply_calibrated_transport(model, variant, config=None):
    target = _target_model(model)
    neck = target.model[9]
    if type(neck) is not HyperACE or list(neck.f) != [4, 6, 8]:
        raise ValueError('Expected native HyperACE layer9 from P3/P4/P5 [4,6,8]')
    before = {name: value.detach().clone() for name, value in target.state_dict().items()}
    parameter_names = set(dict(target.named_parameters()))
    for branch in [neck.branch1, neck.branch2]:
        generator = branch.m.hgnn.edge_generator
        if type(generator) is not AdaHyperedgeGen:
            raise ValueError('Expected an unmodified official generator')
        branch.m.hgnn.edge_generator = CalibratedTransportHyperedgeGen.from_original(generator, variant, config)
    after = target.state_dict()
    assert before.keys() == after.keys()
    assert parameter_names == set(dict(target.named_parameters()))
    assert all(torch.equal(value, after[name]) for name, value in before.items())
    # The official source checkpoint itself has these four weights near
    # +/-6.56e-7, so original biases hide even substantial A changes in AMP.
    # Reset only the four existing matrices, equally for the calibrated no-OT
    # control and every OT strength. Preserve biases and every other tensor.
    # Generate on CPU with a private RNG: no global CPU/CUDA RNG is consumed.
    rng = torch.Generator(device='cpu').manual_seed(INITIALIZATION_SEED)
    initialized = {}
    parameters = dict(target.named_parameters())
    for name in RESET_WEIGHT_NAMES:
        parameter = parameters[name]
        replacement = torch.empty(parameter.shape, dtype=torch.float32, device='cpu')
        torch.nn.init.kaiming_uniform_(replacement, a=math.sqrt(5), generator=rng)
        with torch.no_grad():
            parameter.copy_(replacement.to(device=parameter.device, dtype=parameter.dtype))
        initialized[name] = {
            'shape': list(parameter.shape),
            'initial_tensor_sha256_float32_cpu': hashlib.sha256(replacement.contiguous().numpy().tobytes()).hexdigest(),
            'previous_rms': float(before[name].detach().double().square().mean().sqrt()),
            'reset_rms': float(parameter.detach().double().square().mean().sqrt())}
    after = target.state_dict()
    changed = [name for name, value in before.items() if not torch.equal(value, after[name])]
    assert set(changed) == set(RESET_WEIGHT_NAMES)
    assert all(torch.equal(value, after[name]) for name, value in before.items() if name not in RESET_WEIGHT_NAMES)
    report = {'variant': variant, 'config': variant_config(variant, config),
        'new_parameters': 0, 'new_parameter_names': [], 'new_state_keys': [],
        'allowed_changed_keys': list(RESET_WEIGHT_NAMES), 'actual_changed_keys': changed,
        'all_other_model_state_tensors_identical': True, 'state_tensor_count': len(before),
        'initialization_seed': INITIALIZATION_SEED, 'initialization_changed_tensors': initialized,
        'initialization': 'Only four existing HG projection weight matrices reset to standard nn.Linear Kaiming-uniform distribution with private CPU RNG seed0; original biases and all other tensors preserved',
        'generator_count': 2, 'custom_serialized_class': 'ot_calibrated_hypergraph.CalibratedTransportHyperedgeGen',
        'original_message_passing_unchanged': True, 'neck_class_unchanged': type(neck).__name__,
        'calibration': 'FP32 learned logits; per-edge node centering/std with roundoff guard; original dropout once AFTER calibration',
        'cost_basis': 'Calibrated incidence prior P0=Acal/K',
        'scope': 'Two calibrated AdaHyperedgeGen instances on original fused HyperACE tokens, plus matched initialization repair of four existing HG projection weight matrices; parameter paths/counts and native two-pass formulas unchanged'}
    target.transport_variant = variant
    target.transport_conversion_report = report
    return report


def set_calibrated_capture(model, enabled=True):
    for module in model.modules():
        if isinstance(module, CalibratedTransportHyperedgeGen):
            module.capture_transport = bool(enabled)
            module.clear_transport_cache()


def calibrated_generators(model):
    return {name: module for name, module in model.named_modules()
            if isinstance(module, CalibratedTransportHyperedgeGen)}
