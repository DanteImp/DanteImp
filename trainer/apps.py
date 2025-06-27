from __future__ import annotations

from django.apps import AppConfig
from django.db import connection
from django.db.models.signals import post_migrate


class TrainerConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "trainer"

    def ready(self):
        """PRAGMA journal_mode=WAL ayarını migrate sonrası uygulayın."""

        def enable_wal(sender, **kwargs):
            with connection.cursor() as cur:
                cur.execute("PRAGMA journal_mode=WAL;")
