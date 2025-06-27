import os
import sys
import shutil
import subprocess

# === Proje adı ===
project_name = "ABS_AI"

# === Ana klasörler ===
BASE_DIR = "/home/toyota/Desktop/yolov7_proje"
YOLOV7_DIR = os.path.join(BASE_DIR, "yolov7")
WEIGHTS_PATH = os.path.join(BASE_DIR, "trainOutput", project_name, "weights", "best.pt")
SOURCE_PATH = os.path.join(BASE_DIR, "auto_label", "unlabelled_images")
LABEL_OUTPUT = os.path.join(BASE_DIR, "auto_label", "outputs", "labels")
DRAWN_OUTPUT = os.path.join(BASE_DIR, "auto_label", "outputs", "images")

# === Komut oluştur ===
detect_command = [
    "python3",
    os.path.join(YOLOV7_DIR, "detect.py"),
    "--weights", WEIGHTS_PATH,
    "--source", SOURCE_PATH,
    "--conf", "0.40",
    "--img", "640",
    "--save-txt"
]

# === Detect.py çalıştır ===
print("🚀 YOLOv7 tespiti başlatılıyor...")
subprocess.run(detect_command, check=True)
print("✅ Tespit tamamlandı.\n")

# === Son oluşan klasörü bul ===
runs_dir = os.path.join(BASE_DIR, "runs", "detect")
all_exps = [f for f in os.listdir(runs_dir) if f.startswith("exp")]
latest_exp = sorted(all_exps, key=lambda x: os.path.getctime(os.path.join(runs_dir, x)))[-1]
latest_exp_path = os.path.join(runs_dir, latest_exp)

# === Etiketleri kopyala ===
labels_src = os.path.join(latest_exp_path, "labels")
if os.path.isdir(labels_src):
    os.makedirs(LABEL_OUTPUT, exist_ok=True)
    for f in os.listdir(labels_src):
        shutil.copy(os.path.join(labels_src, f), os.path.join(LABEL_OUTPUT, f))
    print(f"📂 Etiket dosyaları kopyalandı: {LABEL_OUTPUT}")
else:
    print("⚠️ Etiket klasörü bulunamadı.")

# === Görselleri kopyala ===
os.makedirs(DRAWN_OUTPUT, exist_ok=True)
for f in os.listdir(latest_exp_path):
    if f.lower().endswith((".jpg", ".jpeg", ".png")):
        shutil.copy(os.path.join(latest_exp_path, f), os.path.join(DRAWN_OUTPUT, f))
print(f"🖼️ Görseller kopyalandı: {DRAWN_OUTPUT}")

# === İsteğe bağlı: runs klasörünü temizle ===
# shutil.rmtree(runs_dir)

print("\n✅ Auto-label detect süreci tamamlandı.")

