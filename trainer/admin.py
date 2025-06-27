"""trainer.admin – Modelleri Django Admin'e kaydeder."""

from __future__ import annotations

from django.contrib import admin, messages
from django.http import HttpRequest

from .models import Project, Training, TrainingResult, TensorboardResult

# -----------------------------------------------------------------------------
#  Inline'lar
# -----------------------------------------------------------------------------


class TrainingInline(admin.TabularInline):
    model = Training
    extra = 0
    readonly_fields = ("status", "started_at", "finished_at")


class TrainingResultInline(admin.StackedInline):
    model = TrainingResult
    extra = 0


# -----------------------------------------------------------------------------
#  Admin action'ları
# -----------------------------------------------------------------------------


def regenerate_tensorboard(modeladmin: admin.ModelAdmin, request: HttpRequest, queryset):
    """Seçili eğitimler için TensorBoard özetlerini yeniden oluştur."""
    for tr in queryset:
        # işlemi fonksiyon içinde yap → modül importunda veritabanına dokunulmaz
        TensorboardResult.objects.filter(training=tr).delete()
        TensorboardResult.objects.create_from_training(tr)  # model method
    messages.success(request, f"{queryset.count()} eğitim yeniden işlendi ✅")


regenerate_tensorboard.short_description = "Seçili eğitimlerin TensorBoard çıktısını yenile"


# -----------------------------------------------------------------------------
#  ModelAdmin'ler
# -----------------------------------------------------------------------------


@admin.register(Project)
class ProjectAdmin(admin.ModelAdmin):
    list_display = ("name", "status", "created_at")
    list_filter = ("status",)
    inlines = (TrainingInline,)


@admin.register(Training)
class TrainingAdmin(admin.ModelAdmin):
    list_display = ("id", "project", "status", "epochs", "batch_size", "created_at")
    list_filter = ("status", "created_at")
    actions = (regenerate_tensorboard,)
    inlines = (TrainingResultInline,)


@admin.register(TensorboardResult)
class TensorboardResultAdmin(admin.ModelAdmin):
    list_display = ("id", "training", "created_at")

# ---------------  SON  -------------------------------------------------------