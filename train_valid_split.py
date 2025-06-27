import os
import shutil
import argparse
import yaml
from sklearn.model_selection import train_test_split
from pathlib import Path

# Ana dizin
BASE_DIR = Path(__file__).resolve().parent
YOLOV7_CONFIG_SRC = BASE_DIR / "yolov7" / "cfg" / "training" / "yolov7.yaml"
DATA_CONFIG_SRC = BASE_DIR / "yolov7" / "data" / "coco.yaml"

def create_project(project_name, ratio, class_names, dataset_dir):
    project_dir = BASE_DIR / "projects" / project_name

    # Kopyalama
    os.makedirs(project_dir, exist_ok=True)
    shutil.copy(YOLOV7_CONFIG_SRC, project_dir / "yolov7.yaml")
    shutil.copy(DATA_CONFIG_SRC, project_dir / "data.yaml")

    print(f"✅ Proje klasörü oluşturuldu: {project_dir}")
    print(f"📁 Veri klasörü: {dataset_dir}")

    split_yolo_dataset(dataset_dir, project_dir, float(ratio))
    update_yaml_files(project_dir, class_names)

def split_yolo_dataset(dataset_dir, project_dir, train_ratio=0.8):
    images_train_dir = project_dir / 'images/train'
    images_val_dir   = project_dir / 'images/val'
    labels_train_dir = project_dir / 'labels/train'
    labels_val_dir   = project_dir / 'labels/val'

    for d in [images_train_dir, images_val_dir, labels_train_dir, labels_val_dir]:
        os.makedirs(d, exist_ok=True)

    all_images = [f for f in os.listdir(dataset_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not all_images:
        print("❌ Kaynakta görsel yok.")
        return

    train_files, val_files = train_test_split(all_images, train_size=train_ratio, random_state=42)

    def copy_files(files, img_dst, lbl_dst):
        for img in files:
            shutil.copy2(os.path.join(dataset_dir, img), img_dst)
            label = os.path.splitext(img)[0] + '.txt'
            src_lbl = os.path.join(dataset_dir, label)
            if os.path.exists(src_lbl):
                shutil.copy2(src_lbl, lbl_dst)
            else:
                print(f"⚠️ Etiket bulunamadı: {label}")

    copy_files(train_files, images_train_dir, labels_train_dir)
    copy_files(val_files, images_val_dir, labels_val_dir)

    print(f"📊 Eğitim: {len(train_files)}, Doğrulama: {len(val_files)}")

def update_yaml_files(project_dir, class_names):
    yolov7_yaml_path = project_dir / "yolov7.yaml"
    data_yaml_path   = project_dir / "data.yaml"

    # ➤ Sadece 'nc:' satırını değiştir, diğerlerini koru
    with open(yolov7_yaml_path, 'r', encoding="utf-8") as f:
        lines = f.readlines()

    with open(yolov7_yaml_path, 'w', encoding="utf-8") as f:
        for line in lines:
            if line.strip().startswith("nc:"):
                f.write(f"nc: {len(class_names)}\n")
            else:
                f.write(line)

    # ➤ data.yaml yeniden oluştur
    train_path = project_dir / "images/train"
    val_path   = project_dir / "images/val"
    with open(data_yaml_path, 'w', encoding="utf-8") as f:
        f.write(f"train: {train_path}/\n")
        f.write(f"val: {val_path}/\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write("names: [" + ", ".join(f"'{c}'" for c in class_names) + "]\n")

    print("✅ YAML dosyaları güncellendi.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv7 Proje Kurulumu")
    parser.add_argument("project_name", type=str, help="Proje adı")
    parser.add_argument("ratio", type=float, help="Eğitim oranı (örn: 0.9)")
    parser.add_argument("class_names", nargs='+', help="Sınıf isimleri (boşlukla)")
    parser.add_argument("--dataset_dir", type=str, default=str(BASE_DIR / "data"), help="Kaynak veri yolu")
    args = parser.parse_args()

    create_project(args.project_name, args.ratio, args.class_names, args.dataset_dir)

