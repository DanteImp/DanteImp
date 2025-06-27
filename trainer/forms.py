"""trainer.forms – Kullanıcıya sunulan basit Django form'ları."""

from __future__ import annotations

from typing import List

from django import forms

from .models import Project, Training


class ProjectForm(forms.ModelForm):
    class Meta:
        model = Project
        fields = [
            "name",
            "dataset_archive",
            "split_ratio",
            "class_names_raw",  # textarea – satır başına sınıf adı
        ]
        widgets = {
            "class_names_raw": forms.Textarea(attrs={"rows": 4}),
        }

    def clean_class_names_raw(self):
        raw: str = self.cleaned_data["class_names_raw"]
        lst: List[str] = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        if not lst:
            raise forms.ValidationError("En az bir sınıf adı girilmelidir.")
        return json.dumps(lst, ensure_ascii=False)


class TrainingForm(forms.ModelForm):
    """Belirli proje seçilerek eğitim parametreleri alınır."""

    project = forms.ModelChoiceField(queryset=Project.objects.none())  # queryset boş başlıyor

    class Meta:
        model = Training
        fields = ["project", "epochs", "batch_size"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # RuntimeWarning'i önlemek için queryset'i burada ayarla
        self.fields["project"].queryset = Project.objects.filter(status="ready")