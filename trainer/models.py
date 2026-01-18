from pathlib import Path

from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models
from django.utils.text import slugify

BASE_DIR = Path(__file__).resolve().parent.parent


# ───────────────────────── PROJECT ─────────────────────────
class Project(models.Model):
    STATUS_CHOICES = [
        ("Idle", "Idle"), ("started", "Started"), ("done", "Done"),
        ("error", "Error"), ("missing", "Missing")
    ]

    name          = models.CharField(max_length=120, unique=True)
    process_name  = models.CharField(max_length=255, default="Unnamed Process")
    class_names   = models.TextField(help_text="Her satıra bir sınıf ismi yazın.")
    train_ratio   = models.FloatField(default=0.8,
                                      validators=[MinValueValidator(0.0), MaxValueValidator(1.0)])

    status        = models.CharField(max_length=50, choices=STATUS_CHOICES, default="Idle")
    description   = models.TextField(blank=True, null=True)
    created_at    = models.DateTimeField(auto_now_add=True)
    updated_at    = models.DateTimeField(auto_now=True)

    # yardımcı
    def folder_path(self):
        return BASE_DIR / "projects" / self.name

    @property
    def project_dir(self):
        return self.folder_path()

    @property
    def slug(self):
        return slugify(self.name)

    def __str__(self):
        return f"{self.process_name} – {self.name}"


# ──────────────────────── TRAINING (ANA) ───────────────────────
class Training(models.Model):
    STATUS_CHOICES = [
        ("running", "Running"), ("done", "Done"),
        ("error", "Error"),     ("queued", "Queued"),
    ]

    # form alanları
    project       = models.ForeignKey(Project, on_delete=models.CASCADE, related_name="trainings")
    img_size      = models.PositiveIntegerField(default=640)
    epochs        = models.PositiveIntegerField(default=100)
    batch_size    = models.PositiveIntegerField(default=16)
    device        = models.CharField(max_length=10, default="0")       # GPU id
    weights_path  = models.CharField(max_length=255, default="yolov7/yolov7.pt")
    hyp_path      = models.CharField(max_length=255, default="yolov7/data/hyp.scratch.custom.yaml")
    noautoanchor  = models.BooleanField(default=False)
    extra_args    = models.TextField(blank=True, null=True)
    description   = models.TextField(blank=True, null=True)

    # izleme alanları
    start_time        = models.DateTimeField(blank=True, null=True)
    end_time          = models.DateTimeField(blank=True, null=True)
    current_epoch     = models.CharField(max_length=20, blank=True, null=True)
    progress_percent  = models.CharField(max_length=10, blank=True, null=True)
    status            = models.CharField(max_length=20, choices=STATUS_CHOICES, default="queued")
    duration_min      = models.CharField(max_length=20, blank=True, null=True)
    log_file_path     = models.CharField(max_length=512, blank=True, null=True)
    output_dir        = models.CharField(max_length=512, blank=True, null=True)

    # opsiyonel anchor raporu
    anchor_report = models.TextField(blank=True, null=True)
    anchors_json  = models.TextField(blank=True, null=True)

    created_at    = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Training #{self.id} – {self.project.name}"


# ──────────────── TRAINING RESULT (Özet + Sistem) ───────────────
class TrainingResult(models.Model):
    training = models.OneToOneField(Training, on_delete=models.CASCADE, related_name="result")

    # dataset sayıları
    image_count_train = models.IntegerField(default=0)
    image_count_val   = models.IntegerField(default=0)
    empty_image_count = models.IntegerField(default=0)
    num_workers       = models.IntegerField(default=0)

    # model / donanım
    weight_transfer_info = models.CharField(max_length=50, default="0/0")
    gflops          = models.FloatField(default=0.0)
    layer_count     = models.IntegerField(default=0)
    parameter_count = models.BigIntegerField(default=0)
    gpu_memory_usage = models.CharField(max_length=20, default="0G")

    # özet metrikler
    precision  = models.FloatField(null=True, blank=True)
    recall     = models.FloatField(null=True, blank=True)
    mAP50      = models.FloatField(null=True, blank=True)
    mAP50_95   = models.FloatField(null=True, blank=True)

    # loss’lar (son epoch)
    loss_box = models.FloatField(null=True, blank=True)
    loss_obj = models.FloatField(null=True, blank=True)
    loss_cls = models.FloatField(null=True, blank=True)

    # custom-best
    custom_best_model_epoch  = models.IntegerField(null=True, blank=True)
    custom_best_model_score  = models.FloatField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Result – Train #{self.training.id}"


# ───────────────────────── TENSORBOARD RAW ─────────────────────
class TensorboardResult(models.Model):
    training        = models.ForeignKey(Training, on_delete=models.CASCADE,
                                        related_name="tensorboard_entries")
    training_result = models.OneToOneField(TrainingResult, on_delete=models.CASCADE,
                                           related_name="tensorboard")

    train_box_loss = models.FloatField(default=0.0, null=True, blank=True)
    train_obj_loss = models.FloatField(default=0.0, null=True, blank=True)
    train_cls_loss = models.FloatField(default=0.0, null=True, blank=True)
    val_box_loss   = models.FloatField(default=0.0, null=True, blank=True)
    val_obj_loss   = models.FloatField(default=0.0, null=True, blank=True)
    val_cls_loss   = models.FloatField(default=0.0, null=True, blank=True)

    precision    = models.FloatField(default=0.0, null=True, blank=True)
    recall       = models.FloatField(default=0.0, null=True, blank=True)
    map_0_5      = models.FloatField(default=0.0, null=True, blank=True)
    map_0_5_0_95 = models.FloatField(default=0.0, null=True, blank=True)

    lr0 = models.FloatField(default=0.0, null=True, blank=True)
    lr2 = models.FloatField(default=0.0, null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Tensorboard – Train #{self.training.id}"
