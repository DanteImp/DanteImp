import re
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator

# TensorBoard etiket → model alanı haritası
TB_KEY_MAP = {
    "Precision":       "precision",
    "Recall":          "recall",
    "mAP@0.5":         "map_0_5",
    "mAP@0.5:0.95":    "map_0_5_0_95",
}


def parse_yolo_log(log_path: str):
    """
    YOLO eğitim çıktısı (stdout) .log dosyasından loss ve LR değerlerini ayıklar.
    Log dosyası yoksa boş sözlük döner.
    """
    stats: dict[str, float] = {}

    if not log_path:
        return stats
    path = Path(log_path)
    if not path.exists():
        return stats

    tag_to_field = {
        "train/box_loss": "train_box_loss",
        "train/obj_loss": "train_obj_loss",
        "train/cls_loss": "train_cls_loss",
        "val/box_loss":   "val_box_loss",
        "val/obj_loss":   "val_obj_loss",
        "val/cls_loss":   "val_cls_loss",
        "lr/0":           "lr0",
        "lr/2":           "lr2",
    }

    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            for tag, field in tag_to_field.items():
                if tag in line:
                    m = re.search(rf"{tag}[:=]\s*([\d\.Ee+-]+)", line)
                    if m:
                        stats[field] = float(m.group(1))
    return stats


def tensorboard_last_metrics(runs_dir):
    """
    Verilen 'runs/' klasöründen (TensorBoard) Precision/Recall/mAP
    değerlerinin son kaydını döndürür. Klasör yoksa boş sözlük.
    """
    scalars: dict[str, float] = {}

    if not runs_dir:
        return scalars
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        return scalars

    # en yeni alt klasörü al
    subdirs = sorted(runs_dir.glob("*"), key=lambda p: p.stat().st_mtime)
    if not subdirs:
        return scalars

    ea = event_accumulator.EventAccumulator(subdirs[-1])
    ea.Reload()

    for tb_tag, field in TB_KEY_MAP.items():
        if tb_tag in ea.Tags().get("scalars", []):
            scalars[tb_tag] = ea.Scalars(tb_tag)[-1].value

    return scalars
