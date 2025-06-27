#!/usr/bin/env python3
"""Eğitim logunu satır satır parse eden yardımcı fonksiyonlar."""
import re
from datetime import datetime

# ==================== sabit REGEX yardımcıları ====================
_FLOAT = r"[\d\.Ee+-]+"

def _last_match(pattern: str, text: str):
    matches = list(re.finditer(pattern, text))
    return matches[-1] if matches else None


# ==================== temel parser fonksiyonları ==================
def parse_training_params(txt: str) -> dict:
    pat = {
        "batch_size": r"Batch Size\s*:\s*(\d+)",
        "img_size":   r"Image Size\s*:\s*(\d+)",
        "epochs":     r"Epochs\s*:\s*(\d+)",
        "device":     r"Device\s*:\s*(\d+)",
        "weights_path": r"Weights\s*:\s*(.+)",
        "hyp_path":     r"Hyperparams\s*:\s*(.+)",
        "output_dir":   r"Output Dir\s*:\s*(.+)",
        "extra_args":   r"Extra Args\s*:\s*(.+)",
    }
    return {k: (m.group(1).strip() if (m := re.search(p, txt)) else None) for k, p in pat.items()}


def parse_num_workers(txt: str) -> dict:
    m = re.search(r"workers=(\d+)", txt)
    return {"num_workers": int(m.group(1))} if m else {}


def parse_dataset_counts(txt: str) -> dict:
    d = {}
    if (m := re.search(r"train.*?(\d+)\s*found.*?(\d+)\s*empty", txt)):
        d["image_count_train"] = int(m.group(1))
        d["empty_image_count"] = int(m.group(2))
    if (m := re.search(r"val.*?(\d+)\s*found", txt)):
        d["image_count_val_parsed"] = int(m.group(1))
    return d


def parse_save_dir(txt: str) -> dict:
    m = re.search(r"save_dir='([^']+)'", txt)
    return {"save_dir": m.group(1)} if m else {}


def parse_model_summary(txt: str) -> dict:
    m = re.search(r"Model Summary:\s*(\d+)\s*layers,\s*(" + _FLOAT + r")\s*parameters.*?(" + _FLOAT + r")\s*GFLOPS", txt)
    if not m:
        return {}
    return {
        "layer_count": int(m.group(1)),
        "parameter_count": float(m.group(2)),
        "gflops": float(m.group(3)),
    }


def parse_weight_transfer_info(txt: str) -> dict:
    m = re.search(r"Transferred\s+(\d+/\d+)\s+items", txt)
    return {"weight_transfer_info": m.group(1)} if m else {}


def parse_epoch_losses(txt: str) -> dict:
    m = _last_match(rf"(\d+)/(\d+)\s+{_FLOAT}G?\s+({_FLOAT})\s+({_FLOAT})\s+({_FLOAT})\s+({_FLOAT})", txt)
    if not m:
        return {}
    cur, tot = int(m.group(1)), int(m.group(2))
    return {
        "epoch": cur,
        "gpu_mem": m.group(3),
        "box_loss": float(m.group(4)),
        "obj_loss": float(m.group(5)),
        "cls_loss": float(m.group(6)),
        "total_loss": float(m.group(7)),
        "progress_percent": round(cur / tot * 100, 2),
    }


def parse_val_summary(txt: str) -> dict:
    m = _last_match(rf"all\s+(\d+)\s+(\d+)\s+({_FLOAT})\s+({_FLOAT})\s+({_FLOAT})\s+({_FLOAT})", txt)
    if not m:
        return {}
    return {
        "image_count_val": int(m.group(1)),
        "label_count_val": int(m.group(2)),
        "precision": float(m.group(3)),
        "recall": float(m.group(4)),
        "mAP50": float(m.group(5)),
        "mAP50_95": float(m.group(6)),
    }


def parse_best_model_info(txt: str) -> dict:
    m = _last_match(r"Custom best model updated at epoch (\d+) \(score=(" + _FLOAT + r")\)", txt)
    return {"custom_best_model_epoch": int(m.group(1)), "custom_best_model_score": float(m.group(2))} if m else {}


def parse_times(txt: str) -> dict:
    d = {}
    ms = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - === Eğitim başlatıldı", txt)
    me = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - ✅ Eğitim tamamlandı", txt)
    if ms:
        d["start_time"] = ms.group(1)
    if me:
        d["end_time"] = me.group(1)
    if ms and me:
        t1, t2 = (datetime.strptime(x, "%Y-%m-%d %H:%M:%S") for x in (ms.group(1), me.group(1)))
        d["duration_min"] = round((t2 - t1).total_seconds() / 60, 2)
        d["status"] = "done"
    else:
        d["duration_min"] = None
        d["status"] = "running"
    return d
