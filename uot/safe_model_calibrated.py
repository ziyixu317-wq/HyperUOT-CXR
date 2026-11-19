"""Restricted loader for the explicitly calibrated generator-only experiment."""
import builtins
import hashlib
import importlib
import importlib.util
import io
import pickletools
import sys
import zipfile
from pathlib import Path

import torch
import safe_model_base as baseline

PROJECT = baseline.PROJECT
OFFICIAL_SHA256 = baseline.OFFICIAL_SHA256
_RAW_TORCH_LOAD = baseline._RAW_TORCH_LOAD
# Do not admit the cancelled four-class OT implementation in this experiment.
EXPECTED = {name for name in baseline.EXPECTED if not name.startswith('ot_hypergraph ')}
EXPECTED.add('ot_calibrated_hypergraph CalibratedTransportHyperedgeGen')
AUDITED_TRANSPORT_SOURCE = Path(__file__).resolve().parent / "ot_transport_hypergraph.py"
AUDITED_TRANSPORT_SOURCE_SHA256 = "6b6f62892e3e6e5341fcba72cc507c250496e1f2c67fc5a392a34740fccc9456"  # docstring-only change vs. trained revision 718b342f...
AUDITED_CALIBRATED_SOURCE = Path(__file__).resolve().parent / "ot_calibrated_hypergraph.py"
# Reviewed four-matrix matched-initialization repair, with unchanged solvers.
AUDITED_CALIBRATED_SOURCE_SHA256 = "1e733738586892b16c652d46ee6fccb519ab59d479d0d9e2b5171ae43ba75806"  # docstring-only change vs. trained revision 264ab4f7...


def validate_calibrated_conversion(report):
    expected = {f'model.9.{branch}.m.hgnn.{projection}.0.weight'
                for branch in ('branch1', 'branch2') for projection in ('edge_proj', 'node_proj')}
    assert report['new_parameters'] == 0
    assert not report['new_parameter_names'] and not report['new_state_keys']
    assert set(report['allowed_changed_keys']) == expected
    assert set(report['actual_changed_keys']) == expected
    assert report['all_other_model_state_tensors_identical']
    assert report['initialization_seed'] == 0
    assert set(report['initialization_changed_tensors']) == expected
    for item in report['initialization_changed_tensors'].values():
        assert len(item['shape']) == 2 and item['reset_rms'] > 0
        assert len(item['initial_tensor_sha256_float32_cpu']) == 64
    assert report['original_message_passing_unchanged']
    return report


def _prepare(module_name, path, expected):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError(f'Unreviewed {module_name} source revision: {digest}')
    previous = sys.modules.get(module_name)
    if previous is not None:
        if Path(previous.__file__).resolve() != path.resolve():
            raise ValueError(f'{module_name} imported from an unexpected path')
        return previous
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def prepare_audited_calibrated_module():
    # The legacy module supplies reviewed pure solvers; its old serialized
    # generator class is deliberately not allowed in calibrated checkpoints.
    _prepare('ot_transport_hypergraph', AUDITED_TRANSPORT_SOURCE, AUDITED_TRANSPORT_SOURCE_SHA256)
    return _prepare('ot_calibrated_hypergraph', AUDITED_CALIBRATED_SOURCE, AUDITED_CALIBRATED_SOURCE_SHA256)


def load_checkpoint(path, *args, **kwargs):
    if not isinstance(path, (str, Path, io.BytesIO)):
        raise TypeError(f'Unreviewed checkpoint input type {type(path)}')
    position = path.tell() if isinstance(path, io.BytesIO) else None
    try:
        with zipfile.ZipFile(path) as archive:
            name = next(n for n in archive.namelist() if n.endswith('/data.pkl') or n == 'data.pkl')
            found = set()
            for opcode, value, _ in pickletools.genops(archive.read(name)):
                if opcode.name in ('STACK_GLOBAL', 'EXT1', 'EXT2', 'EXT4'):
                    raise ValueError(f'Unreviewed pickle opcode: {opcode.name}')
                if opcode.name == 'GLOBAL':
                    found.add(value)
        unexpected = found - EXPECTED
        if unexpected:
            raise ValueError(f'Unreviewed checkpoint globals: {sorted(unexpected)}')
    finally:
        if position is not None:
            path.seek(position)
    if 'ot_calibrated_hypergraph CalibratedTransportHyperedgeGen' in found:
        prepare_audited_calibrated_module()
    allowed = []
    for name in sorted(found):
        module, symbol = name.split(' ')
        allowed.append(getattr(builtins if module == '__builtin__' else importlib.import_module(module), symbol))
    kwargs['weights_only'] = True
    with torch.serialization.safe_globals(allowed):
        return _RAW_TORCH_LOAD(path, *args, **kwargs)


def install_restricted_loader():
    import numpy as np
    if not hasattr(np, 'trapz'):
        np.trapz = np.trapezoid
    torch.load = load_checkpoint
    import ultralytics.utils.patches as patches
    import ultralytics.nn.tasks as tasks
    import ultralytics.nn.modules.block as block
    patches.torch_load = load_checkpoint
    if hasattr(tasks, 'torch_load'):
        tasks.torch_load = load_checkpoint
    block.USE_FLASH_ATTN = False
    return load_checkpoint
