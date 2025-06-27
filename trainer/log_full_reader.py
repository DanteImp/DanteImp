#!/usr/bin/env python3
"""
Eğitim logunu (5 sn döngüyle) okuyup Trainer DB’sine
insert / update yapan script.
Kullanım:   python trainer/log_full_reader.py --training_id <id>
"""
import argparse, os, sys, time
from pathlib import Path

# Django setup
BASE = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "yolov7_panel.settings")

import django
django.setup()

from django.db import transaction
from trainer.models import Training, TrainingResult
from trainer.log_parser import *

# ─────────────────────── argument ───────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--training_id", type=int, required=True, help="Training tablosundaki id")
args = ap.parse_args()

# ----------------------------------------------------------------
training = Training.objects.select_related("project").get(id=args.training_id)
project_name = training.project.name

# en güncel log dosyasını bul
logs_dir = BASE / "logs"
log_files = sorted(logs_dir.glob(f"{project_name}*.log"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
if not log_files:
    print("❌ Log dosyası bulunamadı")
    sys.exit(1)
log_path = log_files[0]
training.log_file_path = str(log_path)
training.save(update_fields=["log_file_path"])
print(f"📂 İzlenen log: {log_path.name}")

# flag’ler
first_insert_done = bool(training.result_id)

# ----------------------------------------------------------------
while True:
    with log_path.open("r", encoding="utf-8", errors="ignore") as fh:
        txt = fh.read()

    ep      = parse_epoch_losses(txt)
    vs      = parse_val_summary(txt)
    times   = parse_times(txt)
    best    = parse_best_model_info(txt)
    tr_par  = parse_training_params(txt)
    nwork   = parse_num_workers(txt)
    dscnt   = parse_dataset_counts(txt)
    savedir = parse_save_dir(txt)
    modelsum= parse_model_summary(txt)
    wtrans  = parse_weight_transfer_info(txt)

    cur_ep = ep.get("epoch", 0)

    # =============== 1) başlangıç insert ==================
    if not first_insert_done and cur_ep == 0:
        with transaction.atomic():
            # Training ana tablo
            training.batch_size   = tr_par.get("batch_size")
            training.img_size     = tr_par.get("img_size")
            training.epochs       = tr_par.get("epochs")
            training.device       = tr_par.get("device")
            training.output_dir   = tr_par.get("output_dir")
            training.start_time   = times.get("start_time")
            training.status       = "running"
            training.save()

            # TrainingResult
            TrainingResult.objects.create(
                training              = training,
                image_count_train     = dscnt.get("image_count_train"),
                image_count_val       = dscnt.get("image_count_val_parsed"),
                empty_image_count     = dscnt.get("empty_image_count"),
                num_workers           = nwork.get("num_workers"),
                gflops                = modelsum.get("gflops"),
                layer_count           = modelsum.get("layer_count"),
                parameter_count       = modelsum.get("parameter_count"),
                weight_transfer_info  = wtrans.get("weight_transfer_info"),
            )
        first_insert_done = True
        print("✅ İlk insert tamam")

    # =============== 2) eğitim sırasında update ==============
    if cur_ep >= 1:
        with transaction.atomic():
            # Training tablosu
            training.current_epoch    = cur_ep
            training.progress_percent = ep.get("progress_percent")
            training.status           = times.get("status")
            training.save(update_fields=[
                "current_epoch", "progress_percent", "status"
            ])

            # TrainingResult tablosu
            res = training.result
            res.gpu_memory_usage      = ep.get("gpu_mem")
            res.loss_box              = ep.get("box_loss")
            res.loss_obj              = ep.get("obj_loss")
            res.loss_cls              = ep.get("cls_loss")
            res.precision             = vs.get("precision")
            res.recall                = vs.get("recall")
            res.mAP50                 = vs.get("mAP50")
            res.mAP50_95              = vs.get("mAP50_95")
            res.custom_best_model_epoch = best.get("custom_best_model_epoch")
            res.custom_best_model_score = best.get("custom_best_model_score")
            res.save()
        print(f"🔄 Epoch {cur_ep} güncellendi")

    # =============== 3) eğitim bitti ========================
    if times.get("status") == "done":
        training.end_time    = times.get("end_time")
        training.duration_min= times.get("duration_min")
        training.status      = "done"
        training.save(update_fields=["end_time", "duration_min", "status"])
        print("🏁 Eğitim tamamlandı – döngü sonlandırıldı")
        break

    time.sleep(5)
