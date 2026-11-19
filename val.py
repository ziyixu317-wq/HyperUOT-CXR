"""Evaluate a trained UOT checkpoint on the 3000-image validation split.

Usage:  python val.py [--weights results/best.pt] [--device 0]
Settings match the trainer's final validation (batch 32, rect, conf 0.001, iou 0.7, FP32).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import common
from common import ROOT, NAMES, runtime_data_yaml, setup_loader
setup_loader()
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--weights', type=Path, default=ROOT / 'results' / 'best.pt')
    ap.add_argument('--device', default='0')
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--project', type=Path, default=ROOT / 'runs')
    ap.add_argument('--name', default='val_uot')
    a = ap.parse_args()
    data = runtime_data_yaml(a.project)
    model = YOLO(str(a.weights))
    m = model.val(data=str(data), imgsz=640, batch=a.batch, device=a.device, split='val', iou=0.7,
                  max_det=300, half=False, plots=True, project=str(a.project), name=a.name)
    metrics = {k: float(v) for k, v in m.results_dict.items()}
    per_class = {NAMES[int(c)]: {'AP50': float(m.box.ap50[i]), 'AP50_95': float(m.box.ap[i])}
                 for i, c in enumerate(m.box.ap_class_index)}
    out = Path(m.save_dir) / 'val_metrics.json'
    out.write_text(json.dumps({'weights': str(a.weights), 'metrics': metrics, 'per_class': per_class},
                              indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
