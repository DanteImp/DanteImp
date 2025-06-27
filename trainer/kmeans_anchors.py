import os
import argparse
import numpy as np
from sklearn.cluster import KMeans
import yaml
from tqdm import tqdm
from pathlib import Path

def load_labels(label_dir: str):
    """YOLO formatında (w,h) verilerini yükle"""
    labels = []
    label_dir_train = os.path.join(label_dir, "labels/train")
    
    if not os.path.exists(label_dir_train):
        print(f"❌ {label_dir_train} dizini bulunamadı!")
        return labels

    for file in tqdm(os.listdir(label_dir_train), desc="Etiketler yükleniyor"):
        if file.endswith(".txt"):
            with open(os.path.join(label_dir_train, file), "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        _, _, _, w, h = map(float, parts)
                        labels.append([w, h])
    
    if not labels:
        print("⚠️ Etiket bulunamadı – label klasörünü kontrol et!")
    return np.array(labels)

def print_anchor_report(clusters: list[list[int]], img_size: int = 640):
    blocks = [clusters[i : i + 3] for i in range(0, len(clusters), 3)]  # 3'lü gruplar yapıyoruz

    anchors_txt = "\nanchors:\n"
    for block in blocks:
        anchors_txt += "  - [" + ", ".join([f"{int(w * img_size)},{int(h * img_size)}" for w, h in block]) + "]\n"

    print("\n➡ YOLOv7 config için anchor string:")
    print(anchors_txt)

    return anchors_txt

def kmeans_anchors(label_dir: str, n_clusters: int = 9, img_size: int = 640):
    labels = load_labels(label_dir)
    if not labels.size:
        return []

    print("🔎 Anchor analizi başlatıldı...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(labels)
    anchors = sorted(kmeans.cluster_centers_, key=lambda x: x[0] * x[1])  # alana göre sırala

    # Ortalama IoU hesapla
    ious = []
    for w, h in labels:
        iou = [min(w, anchor[0]) * min(h, anchor[1]) / (w * h + anchor[0] * anchor[1] - min(w, anchor[0]) * min(h, anchor[1])) for anchor in anchors]
        ious.append(max(iou))
    avg_iou = np.mean(ious)

    print_anchor_report(anchors, img_size)
    print(f"\n📈 Ortalama IoU: {avg_iou:.4f}")

    return anchors

def main():
    parser = argparse.ArgumentParser(description="YOLOv7 Anchor Cluster Analizi")
    parser.add_argument("--project_dir", type=str, required=True, help="Proje dizini")
    parser.add_argument("--img_size", type=int, default=640, help="Görsel boyutu (örneğin 640)")
    args = parser.parse_args()

    print(f"\n🔎 Anchor analizi başlatıldı → Etiket klasörü: {args.project_dir}/labels/train")
    anchors = kmeans_anchors(args.project_dir, img_size=args.img_size)

    if not anchors:
        print("⚠️ Anchor analizi yapılamadı. Etiketlerde yeterli çeşitlilik yok olabilir.")
        return


if __name__ == "__main__":
    main()