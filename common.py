"""Shared paths / environment for HyperUOT-CXR (all paths relative to this package)."""
from __future__ import annotations
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / 'config'))
os.environ.update(YOLO_AUTOINSTALL='false', WANDB_MODE='disabled', COMET_MODE='DISABLED',
                  YOLO_OFFLINE='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
import sys
FRAMEWORK = ROOT / 'framework'
UOT = ROOT / 'uot'
sys.path.insert(0, str(UOT)); sys.path.insert(0, str(FRAMEWORK))

import yaml

PRETRAINED = ROOT / 'weights' / 'yolov13l.pt'          # COCO-pretrained L detector (framework weights)
DATA_ROOT = ROOT / 'data' / 'vindr-cxr-17c'
TRAIN_LIST = DATA_ROOT / 'train_list.txt'               # 6748 images actually used for training
VARIANT = 'calibrated_soft_source_rho05_tau2'           # best UOT variant (rho=0.5, tau=2.0)
NAMES = ['肺结节', '片状模糊影', '纤维条索影', '肺部肿块', '胸腔积液', '纵隔增宽', '心脏增大',
         '胸膜增厚', '肺内钙化', '肺不张', '支气管扩张', '肺水肿', '间质性改变', '空气支气管征', '气胸', '肺气肿', '空洞']


def runtime_data_yaml(out_dir: Path) -> Path:
    """Write a dataset yaml with absolute paths for the current machine."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / 'dataset.runtime.yaml'
    path.write_text(yaml.safe_dump({'path': str(DATA_ROOT), 'train': str(TRAIN_LIST), 'val': 'images/val',
                                    'nc': 17, 'names': NAMES}, allow_unicode=True, sort_keys=False), encoding='utf-8')
    return path


def setup_loader():
    """Restricted weights_only checkpoint loader + numpy/flash-attn compat, as in the original run."""
    from safe_model_calibrated import install_restricted_loader, EXPECTED
    EXPECTED.add('ot_calibrated_hypergraph CalibratedTransportHyperedgeGen')
    install_restricted_loader()
