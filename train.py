"""Reproduce the best UOT run: 17-class VinDr-CXR, calibrated soft-source/soft-target OT hypergraph.

Usage:  python train.py [--device 0] [--batch 16] [--workers 4] [--name train_uot]
Same hyper-parameters, seed and model conversion as the original run; scheduler / audit
bookkeeping of the original worker script was removed.
"""
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path

import common
from common import ROOT, PRETRAINED, VARIANT, NAMES, runtime_data_yaml, setup_loader

import numpy as np
import torch
setup_loader()
from ot_calibrated_hypergraph import apply_calibrated_transport, CalibratedTransportHyperedgeGen
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer

RESET = {f'model.9.{b}.m.hgnn.{p}.0.weight' for b in ('branch1', 'branch2') for p in ('edge_proj', 'node_proj')}


def plain(value):
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return type(value)(plain(v) for v in value)
    return value


class UOTDetectionTrainer(DetectionTrainer):
    """Native trainer; after the 17-class model + COCO weights are built, swap in the UOT hyperedge generators."""
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        assert self.data['nc'] == 17 and model.model[-1].nc == 17
        report = apply_calibrated_transport(model, VARIANT)
        assert report['new_parameters'] == 0 and set(report['actual_changed_keys']) == RESET
        model.transport_experiment_variant = VARIANT
        (Path(self.save_dir) / 'conversion.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
        return model

    def save_model(self):
        # Same as original run: store plain-python metrics, drop cached criterion from EMA copy.
        ema = self.ema.ema; sentinel = object()
        criterion = getattr(ema, 'criterion', sentinel); args = getattr(ema, 'args', sentinel)
        m, f, b = self.metrics, self.fitness, self.best_fitness
        self.metrics, self.fitness, self.best_fitness = plain(m), plain(f), plain(b)
        sd = self.optimizer.state_dict; self.optimizer.state_dict = lambda: plain(sd())
        if criterion is not sentinel: ema.criterion = None
        if args is not sentinel: ema.args = plain(args if isinstance(args, dict) else vars(args))
        try: super().save_model()
        finally:
            self.optimizer.state_dict = sd; self.metrics, self.fitness, self.best_fitness = m, f, b
            if criterion is not sentinel: ema.criterion = criterion
            if args is not sentinel: ema.args = args


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--device', default='0')
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--project', type=Path, default=ROOT / 'runs')
    ap.add_argument('--name', default='train_uot')
    ap.add_argument('--epochs', type=int, default=100)
    a = ap.parse_args()
    data = runtime_data_yaml(a.project)
    cfg = dict(data=str(data), epochs=a.epochs, patience=30, batch=a.batch, imgsz=640, device=a.device, workers=a.workers,
        project=str(a.project), name=a.name, exist_ok=False, pretrained=True, optimizer='auto', seed=0,
        deterministic=True, cos_lr=True, close_mosaic=15, amp=True, cache=False, save=True, save_period=-1,
        lr0=.01, lrf=.01, momentum=.937, weight_decay=.0005, warmup_epochs=3., warmup_momentum=.8, warmup_bias_lr=.1,
        box=7.5, cls=.5, dfl=1.5, nbs=64, hsv_h=0., hsv_s=0., hsv_v=.3, degrees=5., translate=.1, scale=.3,
        shear=0., perspective=0., flipud=0., fliplr=0., mosaic=1., mixup=0., copy_paste=0., erasing=0.,
        val=True, split='val', iou=.7, max_det=300, plots=True)
    torch.set_num_threads(4); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    import ultralytics.engine.trainer as trainer_module
    trainer_module.check_amp = lambda _: True   # as in original run (AMP verified separately)
    model = YOLO(str(PRETRAINED))
    t0 = time.time()
    result = model.train(trainer=UOTDetectionTrainer, **cfg)
    final = {k: float(v) for k, v in result.results_dict.items()}
    assert all(math.isfinite(v) for v in final.values())
    out = Path(result.save_dir) / 'final_metrics.json'
    out.write_text(json.dumps({'variant': VARIANT, 'metrics': final, 'hours': (time.time() - t0) / 3600}, indent=2))
    print(json.dumps(final, indent=2))


if __name__ == '__main__':
    main()
