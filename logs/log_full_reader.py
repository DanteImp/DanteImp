#!/usr/bin/env python3

import re
from pathlib import Path
import pandas as pd
from datetime import datetime
import time


def parse_num_workers(log_text: str) -> dict:
    pattern = re.compile(r"workers=(\d+)")
    m = pattern.search(log_text)
    return {"num_workers": m.group(1)} if m else {"num_workers": None}


# ──────────────────────────────
def parse_epoch_losses(log_text: str) -> dict:
    pattern = re.compile(
        r"(\d+)/(\d+)\s+([\d\.]+[GM])\s+([\d\.Ee+-]+)\s+([\d\.Ee+-]+)\s+([\d\.Ee+-]+)\s+([\d\.Ee+-]+)"
    )
    matches = list(pattern.finditer(log_text))
    if matches:
        m = matches[-1]
        current_epoch = int(m.group(1))
        max_epoch = int(m.group(2))
        percent = round((current_epoch / max_epoch) * 100, 2) if max_epoch else None
        return {
            "epoch": m.group(1),
            "gpu_mem": m.group(3),
            "box_loss": m.group(4),
            "obj_loss": m.group(5),
            "cls_loss": m.group(6),
            "total_loss": m.group(7),
            "progress_percent": percent
        }
    return {}


# ──────────────────────────────
def parse_val_summary(log_text: str) -> dict:
    summary = {}
    pattern = re.compile(
        r"all\s+(\d+)\s+(\d+)\s+([\d\.Ee+-]+)\s+([\d\.Ee+-]+)\s+([\d\.Ee+-]+)\s+([\d\.Ee+-]+)"
    )
    matches = list(pattern.finditer(log_text))
    if matches:
        m = matches[-1]  # en son val summary satırını al
        summary = {
            "image_count_val": m.group(1),
            "label_count_val": m.group(2),
            "precision": m.group(3),
            "recall": m.group(4),
            "mAP50": m.group(5),
            "mAP50_95": m.group(6),
        }
    return summary

# ──────────────────────────────
def parse_model_summary(log_text: str) -> dict:
    summary = {}
    pattern = re.compile(
        r"Model Summary:\s*(\d+)\s*layers,\s*([\d\.Ee+-]+)\s*parameters.*?([\d\.Ee+-]+)\s*GFLOPS"
    )
    m = pattern.search(log_text)
    if m:
        summary = {
            "layer_count": m.group(1),
            "parameter_count": m.group(2),
            "gflops": m.group(3),
        }
    return summary

# ──────────────────────────────
def parse_save_dir(log_text: str) -> dict:
    pattern = re.compile(r"save_dir='([^']+)'")
    m = pattern.search(log_text)
    return {"save_dir": m.group(1)} if m else {"save_dir": None}

# ──────────────────────────────
def parse_dataset_counts(log_text: str) -> dict:
    counts = {}
    m_train = re.search(r"train.*?(\d+)\s*found.*?(\d+)\s*empty", log_text)
    if m_train:
        counts["image_count_train"] = m_train.group(1)
        counts["empty_image_count"] = m_train.group(2)
    m_val = re.search(r"val.*?(\d+)\s*found", log_text)
    if m_val:
        counts["image_count_val_parsed"] = m_val.group(1)
    return counts

# ──────────────────────────────
def parse_weight_transfer_info(log_text: str) -> dict:
    pattern = re.compile(r"Transferred\s+(\d+/\d+)\s+items")
    m = pattern.search(log_text)
    return {"weight_transfer_info": m.group(1)} if m else {"weight_transfer_info": None}

# ──────────────────────────────
def parse_times(log_text: str) -> dict:
    from datetime import datetime
    times = {}

    # Start time
    m_start = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - === Eğitim başlatıldı", log_text)
    # End time
    m_end = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - ✅ Eğitim tamamlandı", log_text)

    if m_start:
        times["start_time"] = m_start.group(1)
    if m_end:
        times["end_time"] = m_end.group(1)

    if m_start and m_end:
        # Eğitim tamamlanmış
        t1 = datetime.strptime(m_start.group(1), "%Y-%m-%d %H:%M:%S")
        t2 = datetime.strptime(m_end.group(1), "%Y-%m-%d %H:%M:%S")
        duration_min = round((t2 - t1).total_seconds() / 60, 2)
        times["duration_min"] = duration_min
        times["status"] = "done"
    elif m_start:
        # Eğitim devam ediyor
        t1 = datetime.strptime(m_start.group(1), "%Y-%m-%d %H:%M:%S")
        t2 = datetime.now()
        duration_min = round((t2 - t1).total_seconds() / 60, 2)
        times["duration_min"] = duration_min
        times["status"] = "running"
    else:
        # Başlama zamanı yoksa
        times["duration_min"] = None
        times["status"] = "unknown"

    return times

def parse_best_model_info(log_text: str) -> dict:
    """
    Custom best model updated satırından epoch ve score değerini alır (en son olanı).
    """
    pattern = re.compile(r"Custom best model updated at epoch (\d+) \(score=([\d\.Ee+-]+)\)")
    matches = list(pattern.finditer(log_text))
    if matches:
        m = matches[-1]
        return {
            "custom_best_model_epoch": m.group(1),
            "custom_best_model_score": m.group(2)
        }
    else:
        return {
            "custom_best_model_epoch": None,
            "custom_best_model_score": None
        }


# ──────────────────────────────
if __name__ == "__main__":
    log_dir = Path(__file__).resolve().parent
    log_files = sorted(log_dir.glob("RR_GARNISH_NUT_NEW*.log"), key=lambda x: x.stat().st_mtime, reverse=True)
    log_file = log_files[0] if log_files else None

    if not log_file or not log_file.exists():
        print(f"❌ Uygun log dosyası bulunamadı: {log_dir}")
        exit(1)

    print(f"📂 Takip edilen log dosyası: {log_file.name}")
    while True:
        with log_file.open("r", encoding="utf-8", errors="ignore") as f:
            log_content = f.read()

        # Parse işlemleri
        epoch_losses = parse_epoch_losses(log_content)
        val_summary = parse_val_summary(log_content)
        times = parse_times(log_content)
        best_model_info = parse_best_model_info(log_content)

        current_epoch = epoch_losses.get("epoch") if epoch_losses else None


        # Eğitim parametrelerini ilk loopta çek
 
        num_workers = parse_num_workers(log_content)
        dataset_counts = parse_dataset_counts(log_content)
        save_dir = parse_save_dir(log_content)

        if current_epoch is None or int(current_epoch) == 0:
            # İlk başta sadece eğitim parametrelerini print et
            print("📦 İlk başlangıç parametreleri:")

            print(num_workers)
            print(dataset_counts)
            print(save_dir)

        if current_epoch is not None and int(current_epoch) != 0:
            print("✅ Epoch verileri bulundu, gönderilecek güncelleme:")
            print("🔢 Epoch Losses:")
            print(epoch_losses)
            print("📈 Validation Summary:")
            print(val_summary)
            print("🏆 Best Model Info:")
            print(best_model_info)
            print("⏳ Time Info:")
            print(times)
        else:
            print("⚠️ Epoch henüz başlamamış, sadece zaman, status ve best model info gönderiliyor:")
            print("⏳ Time Info:")
            print(times)

        print("🕒 5 saniye bekleniyor...\n")
        time.sleep(5)
