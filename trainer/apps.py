from django.apps import AppConfig
from django.db import connection
from django.db.models.signals import post_migrate


class TrainerConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "trainer"

    def ready(self):
        # migrate bittikten sonra çalışacak
        def enable_wal(sender, **kwargs):
            with connection.cursor() as cur:
                cur.execute("PRAGMA journal_mode=WAL;")

        post_migrate.connect(enable_wal, sender=self)
