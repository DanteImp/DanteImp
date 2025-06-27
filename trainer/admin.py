"""trainer.admin – Django Admin ayarları."""

from __future__ import annotations

from django.contrib import admin, messages
from django.http import HttpRequest

from .models import Project, Training, TrainingResult, TensorboardResult


class TrainingInline(admin.TabularInline):
    model = Training
    extra = 0
    readonly_fields = ("status", "started_at", "finished_at")


class TrainingResultInline(admin.StackedInline):
    model = TrainingResult
    extra = 0


def regenerate_tensorboard(modeladmin: admin.ModelAdmin, request: HttpRequest, queryset):
    for tr in queryset:
        TensorboardResult.objects.filter(training=tr).delete()
        TensorboardResult.objects.create_from_training(tr)
    messages.success(request, f"{queryset.count()} eğitim yeniden işlendi ✅")


regenerate_tensorboard.short_description = "Seçili eğitimlerin TensorBoard çıktısını yenile"


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
