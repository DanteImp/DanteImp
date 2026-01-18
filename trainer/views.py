"""trainer.views – HTTP endpoint'leri."""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path
from typing import List

from django.conf import settings
from django.contrib import messages
from django.core.paginator import Paginator
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST

from .forms import ProjectForm, TrainingForm
from .models import Project, Training

BASE_DIR = Path(settings.BASE_DIR)


def _split_class_names(raw: str) -> List[str]:
    return [name.strip() for name in raw.splitlines() if name.strip()]


def _run_split_script(project: Project) -> None:
    ratio = project.train_ratio or 0.8
    dataset_root = project.folder_path()
    class_names = _split_class_names(project.class_names)
    python_path = getattr(settings, "VENV_PYTHON", sys.executable)
    args: List[str] = [
        python_path,
        str(BASE_DIR / "train_valid_split.py"),
        project.slug,
        str(ratio),
        *class_names,
        "--dataset_dir",
        str(dataset_root),
    ]
    subprocess.run(args, cwd=BASE_DIR, check=True)


def project_list(request: HttpRequest) -> HttpResponse:
    projects = Project.objects.all().order_by("-created_at")
    return render(request, "trainer/project_list.html", {"projects": projects, "form": ProjectForm()})


def project_create(request: HttpRequest) -> HttpResponse:
    if request.method != "POST":
        return redirect("trainer:project_list")

    form = ProjectForm(request.POST, request.FILES)
    if not form.is_valid():
        projects = Project.objects.all().order_by("-created_at")
        return render(request, "trainer/project_list.html", {"projects": projects, "form": form})

    project: Project = form.save(commit=False)
    project.status = "started"
    project.save()

    try:
        project.folder_path().mkdir(parents=True, exist_ok=True)
        _run_split_script(project)
        project.status = "done"
        project.save(update_fields=["status"])
        messages.success(request, f"{project.name} dataseti hazırlandı ✅")
    except subprocess.CalledProcessError as exc:
        project.status = "error"
        project.save(update_fields=["status"])
        messages.error(request, f"Split betiği hata verdi: {exc}")

    return redirect("trainer:project_list")


def train_start(request: HttpRequest, project_id: int) -> HttpResponse:
    project = get_object_or_404(Project, pk=project_id)
    if request.method == "POST":
        form = TrainingForm(request.POST)
        if form.is_valid():
            training = form.save(commit=False)
            training.project = project
            training.status = "queued"
            training.save()

            cmd = [
                "./train.sh",
                "--cfg",
                str(project.project_dir / "yolov7.yaml"),
                "--data",
                str(project.project_dir / "data.yaml"),
                "--img",
                str(training.img_size),
                "--epochs",
                str(training.epochs),
                "--batch-size",
                str(training.batch_size),
                "--device",
                training.device,
                "--weights",
                training.weights_path,
                "--hyp",
                training.hyp_path,
            ]
            if training.noautoanchor:
                cmd.append("--noautoanchor")
            if training.extra_args:
                cmd.extend(shlex.split(training.extra_args))
            subprocess.Popen(cmd, cwd=BASE_DIR)
            messages.success(request, "Eğitim kuyruğa alındı ▶️")
        else:
            messages.error(request, "Form hatalı; eğitime başlanamadı")

    return redirect("trainer:project_list")


def train_view(request: HttpRequest) -> HttpResponse:
    if request.method == "POST":
        form = TrainingForm(request.POST)
        if form.is_valid():
            training = form.save(commit=False)
            training.status = "queued"
            training.save()

            project = training.project
            cmd = [
                "./train.sh",
                "--cfg",
                str(project.project_dir / "yolov7.yaml"),
                "--data",
                str(project.project_dir / "data.yaml"),
                "--img",
                str(training.img_size),
                "--epochs",
                str(training.epochs),
                "--batch-size",
                str(training.batch_size),
                "--device",
                training.device,
                "--weights",
                training.weights_path,
                "--hyp",
                training.hyp_path,
            ]
            if training.noautoanchor:
                cmd.append("--noautoanchor")
            if training.extra_args:
                cmd.extend(shlex.split(training.extra_args))
            subprocess.Popen(cmd, cwd=BASE_DIR)
            messages.success(request, "Eğitim kuyruğa alındı ▶️")
            return redirect("trainer:training_result_list")
        messages.error(request, "Form hatalı; eğitime başlanamadı")
    else:
        form = TrainingForm()

    return render(request, "trainer/train.html", {"form": form})


def training_result_list(request: HttpRequest) -> HttpResponse:
    trainings = Training.objects.select_related("project").order_by("-created_at")
    paginator = Paginator(trainings, 10)
    page = paginator.get_page(request.GET.get("page"))
    return render(request, "trainer/training_results.html", {"trainings": page})


@require_POST
def update_training(request: HttpRequest) -> JsonResponse:
    training = get_object_or_404(Training, pk=request.POST.get("training_id"))
    updatable_fields = [
        "status",
        "current_epoch",
        "progress_percent",
        "log_file_path",
        "output_dir",
        "duration_min",
        "anchor_report",
        "anchors_json",
    ]
    updates = {field: request.POST[field] for field in updatable_fields if field in request.POST}
    if updates:
        for field, value in updates.items():
            setattr(training, field, value)
        training.save(update_fields=list(updates.keys()))
    return JsonResponse({"ok": True, "updated_fields": list(updates.keys())})
