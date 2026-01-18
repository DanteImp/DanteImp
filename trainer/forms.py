"""trainer.forms – Kullanıcı formları."""

from __future__ import annotations

from typing import List

from django import forms

from .models import Project, Training


class ProjectForm(forms.ModelForm):
    class Meta:
        model = Project
        fields = ["name", "process_name", "class_names", "train_ratio", "description"]
        widgets = {
            "class_names": forms.Textarea(attrs={"rows": 4, "placeholder": "Her satıra bir sınıf adı yazın."}),
            "description": forms.Textarea(attrs={"rows": 3}),
        }

    def clean_class_names(self):
        raw: str = self.cleaned_data["class_names"]
        lst: List[str] = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        if not lst:
            raise forms.ValidationError("En az bir sınıf adı girilmelidir.")
        return "\n".join(lst)


class TrainingForm(forms.ModelForm):
    project = forms.ModelChoiceField(queryset=Project.objects.none())

    class Meta:
        model = Training
        fields = [
            "project",
            "img_size",
            "epochs",
            "batch_size",
            "device",
            "weights_path",
            "hyp_path",
            "noautoanchor",
            "extra_args",
            "description",
        ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["project"].queryset = Project.objects.filter(status="done")
