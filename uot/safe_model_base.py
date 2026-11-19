"""Restricted loader for the audited base detector and this experiment's checkpoints."""
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
_RAW_TORCH_LOAD = torch.load
PROJECT = Path(__file__).resolve().parents[1] / "framework"
sys.path.insert(0, str(PROJECT))
import ultralytics

AUDITED_GLOBALS = '''__builtin__ set
collections OrderedDict
torch FloatStorage
torch HalfStorage
torch LongStorage
torch Size
torch._utils _rebuild_parameter
torch._utils _rebuild_tensor_v2
torch.nn.modules.activation GELU
torch.nn.modules.activation SiLU
torch.nn.modules.batchnorm BatchNorm2d
torch.nn.modules.container ModuleList
torch.nn.modules.container Sequential
torch.nn.modules.conv Conv2d
torch.nn.modules.dropout Dropout
torch.nn.modules.linear Identity
torch.nn.modules.linear Linear
torch.nn.modules.pooling AvgPool2d
torch.nn.modules.upsampling Upsample
ultralytics.nn.modules.block A2C2f
ultralytics.nn.modules.block AAttn
ultralytics.nn.modules.block ABlock
ultralytics.nn.modules.block AdaHGComputation
ultralytics.nn.modules.block AdaHGConv
ultralytics.nn.modules.block AdaHyperedgeGen
ultralytics.nn.modules.block C3AH
ultralytics.nn.modules.block DFL
ultralytics.nn.modules.block DSBottleneck
ultralytics.nn.modules.block DSC3k
ultralytics.nn.modules.block DSC3k2
ultralytics.nn.modules.block DownsampleConv
ultralytics.nn.modules.block FullPAD_Tunnel
ultralytics.nn.modules.block FuseModule
ultralytics.nn.modules.block HyperACE
ultralytics.nn.modules.conv Concat
ultralytics.nn.modules.conv Conv
ultralytics.nn.modules.conv DSConv
ultralytics.nn.modules.conv DWConv
ultralytics.nn.modules.head Detect
ultralytics.nn.tasks DetectionModel
ultralytics.utils IterableSimpleNamespace
ot_hypergraph OTHyperACE
ot_hypergraph OTAdaHGComputation
ot_hypergraph OTAdaHGConv
ot_hypergraph OTAdaHyperedgeGen'''
EXPECTED = set(AUDITED_GLOBALS.splitlines())
OFFICIAL_SHA256 = 'f95ad5bbf3aa80a3df2a28ff4c623582ded6e02bb43d7cea9063c2b248e316cb'
AUDITED_OT_SOURCE = Path(__file__).resolve().parent / "ot_hypergraph.py"  # legacy, not shipped/used
# Filled only after reviewing this exact source revision, not supplied by a checkpoint.
AUDITED_OT_SOURCE_SHA256 = 'be15a3b39adc26514381977eaaffb11bbe848db7ff49e8e04bb8fc19fdc2bef3'


def prepare_audited_ot_module():
    digest = hashlib.sha256(AUDITED_OT_SOURCE.read_bytes()).hexdigest()
    if digest != AUDITED_OT_SOURCE_SHA256:
        raise ValueError(f'OT module source has not been audited at this revision: {digest}')
    existing = sys.modules.get('ot_hypergraph')
    if existing is not None:
        if Path(existing.__file__).resolve() != AUDITED_OT_SOURCE.resolve():
            raise ValueError('OT module was imported from an unexpected path')
        return existing
    spec = importlib.util.spec_from_file_location('ot_hypergraph', AUDITED_OT_SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules['ot_hypergraph'] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop('ot_hypergraph', None)
        raise
    return module


def load_checkpoint(path, *args, **kwargs):
    """Inspect GLOBAL opcodes before restricted unpickling; reject unexpected classes."""
    if not isinstance(path, (str, Path, io.BytesIO)):
        raise TypeError(f'Checkpoint source type not reviewed: {type(path)}')
    saved_position = path.tell() if isinstance(path, io.BytesIO) else None
    try:
        with zipfile.ZipFile(path) as archive:
            name = next(n for n in archive.namelist() if n.endswith('/data.pkl') or n == 'data.pkl')
            globals_found = set()
            for opcode, value, _ in pickletools.genops(archive.read(name)):
                if opcode.name in ('STACK_GLOBAL', 'EXT1', 'EXT2', 'EXT4'):
                    raise ValueError(f'Unreviewed pickle opcode: {opcode.name}')
                if opcode.name == 'GLOBAL':
                    globals_found.add(value)
        unexpected = globals_found - EXPECTED
        if unexpected:
            raise ValueError(f'Unreviewed checkpoint globals: {sorted(unexpected)}')
    finally:
        if saved_position is not None:
            path.seek(saved_position)
    allowed = []
    if any(name.startswith('ot_hypergraph ') for name in globals_found):
        prepare_audited_ot_module()
    for name in sorted(globals_found):
        module, symbol = name.split(' ')
        allowed.append(getattr(builtins if module == '__builtin__' else importlib.import_module(module), symbol))
    kwargs['weights_only'] = True
    with torch.serialization.safe_globals(allowed):
        return _RAW_TORCH_LOAD(path, *args, **kwargs)


def install_restricted_loader():
    import numpy as np
    if not hasattr(np, 'trapz'):
        np.trapz = np.trapezoid  # NumPy 2.4 spelling compatibility, identical integration.
    torch.load = load_checkpoint
    import ultralytics.utils.patches as patches
    patches.torch_load = load_checkpoint
    import ultralytics.nn.tasks as tasks
    if hasattr(tasks, 'torch_load'):
        tasks.torch_load = load_checkpoint
    # Keep the official fallback, which works with the installed CUDA 12.8 runtime.
    import ultralytics.nn.modules.block as block
    block.USE_FLASH_ATTN = False
    return load_checkpoint
