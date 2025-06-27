"""trainer.views – HTTP endpoint'leri."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List

from django.conf import settings
from django.contrib import messages
from django.http import HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404, redirect, render

from .forms import ProjectForm, TrainingForm
from .models import Project, Training

BASE_DIR = Path(settings.BASE_DIR)


def _run_split_script(project: Project) -> None:
    ratio = project.split_ratio or 0.8
    args: List[str] = [
        settings.VENV_PYTHON,
        str(BASE_DIR / "train_valid_split.py"),
        project.slug,
        str(ratio),
        *project.class_names,
        "--dataset_dir",
        str(project.dataset_root),
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
    project.status = "preparing"
    project.save()

    try:
        _run_split_script(project)
        project.status = "ready"
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
                "--epochs",
                str(training.epochs),
            ]
            subprocess.Popen(cmd, cwd=BASE_DIR)
            messages.success(request, "Eğitim kuyruğa alındı ▶️")
        else:
            messages.error(request, "Form hatalı; eğitime başlanamadı")

    return redirect("trainer:project_detail", project_id=project.pk)
