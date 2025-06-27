#!/bin/bash

echo "🚀 YOLOv7 Ortam Kurulumu Başlıyor (Ubuntu 22.04 - NVIDIA A5000)..."

# 1. Sistem güncelleme ve temel kütüphaneler
echo "📦 Gerekli sistem paketleri kuruluyor..."
sudo apt update && sudo apt install -y \
  python3 python3-pip python3-venv git build-essential \
  libgl1 libglib2.0-0 libopencv-dev

# 2. Sanal ortam oluşturuluyor
echo "🧪 Python sanal ortam (yololab7) oluşturuluyor..."
python3 -m venv yololab7
source yololab7/bin/activate

# 3. pip ve PyTorch kurulumu (2.5.1 + CUDA 11.8)
echo "⚙️ pip ve torch kurulumu..."
pip install --upgrade pip setuptools wheel
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118


# 4. YOLOv7 klonlanıyor ve ağırlıklar indiriliyor
echo "📁 YOLOv7 deposu indiriliyor..."
git clone https://github.com/WongKinYiu/yolov7.git
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt -P yolov7/

# ➕ Torch satırlarını requirements.txt içinden yorum satırı yap (manuel silmeye gerek kalmaz!)
sed -i '/^torch/ s/^/# /' yolov7/requirements.txt
sed -i '/^torchvision/ s/^/# /' yolov7/requirements.txt

# 5. Gerekli Python bağımlılıkları yükleniyor
echo "📦 Python bağımlılıkları yükleniyor..."
pip install -r yolov7/requirements.txt

# 6. CUDA kontrolü
echo "🧪 CUDA kontrolü:"
python3 -c "import torch; print('CUDA kullanılabilir mi?', torch.cuda.is_available())"

# 7. Bitir
echo ""
echo "✅ YOLOv7 ortam kurulumu tamamlandı!"
echo "📂 Eğitim için: python3 yolov7/train.py --cfg cfg/training/yolov7.yaml --data data.yaml --weights '' --device 0"
echo "🔍 Tahmin için: python3 yolov7/detect.py --weights yolov7.pt --source path_to_image"

