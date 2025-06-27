#!/bin/bash

echo "🔍 Aktif GPU kullanan Python işlemleri aranıyor..."
PIDS=$(ps -eo pid,cmd --sort=-%mem | grep python | grep -v grep | awk '{print $1}')

if [ -z "$PIDS" ]; then
    echo "✅ GPU kullanan aktif python işlemi bulunamadı."
else
    echo "⚠️ Aşağıdaki PID'ler sonlandırılacak:"
    echo "$PIDS"
    for PID in $PIDS; do
        echo "🧹 PID $PID sonlandırılıyor..."
        kill -9 $PID
    done
    echo "✅ GPU belleği temizlendi (aktif Python süreçleri kapatıldı)."
fi

echo "📊 GPU durumu:"
nvidia-smi
