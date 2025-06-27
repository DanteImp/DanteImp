#!/usr/bin/env bash

# train.sh — YOLOv7 Eğitim Scripti (v3 - Final)
# ---------------------------------------------
# • Eğitim ayarlarını alır (-b -e -p vs.)
# • yolov7/train.py'ye aktarır
# • Kayıt tutar, eksik model indirir, log dosyası üretir
# • Eğitim bitince sesle uyarır

set -euo pipefail

# Yardımcılar
real() { realpath -m "$1"; }
log()  { printf '%(%F %T)T - %s\n' -1 "$1" | tee -a "$LOG_FILE"; }
need() { [[ -n ${!1:-} ]] || { echo "Eksik parametre: --${1,,} gerekli"; exit 1; }; }

auto_dl() {
  local dst="$1" url="https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt"
  [[ -f $dst ]] && return 0
  log "Pretrained model eksik → yolov7.pt indiriliyor..."
  mkdir -p "$(dirname "$dst")"
  command -v wget &>/dev/null && wget -q --show-progress -O "$dst" "$url" || \
    curl -L "$url" -o "$dst"
  [[ -f $dst ]] || { log "Model indirme başarısız"; exit 1; }
}

# Argümanları ayrıştır
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    -b|--batch-size)   BATCH=$2; shift 2;;
    -p|--project-name) PROJ=$2; shift 2;;
    -s|--img-size)     IMG=$2; shift 2;;
    -e|--epochs)       EPOCHS=$2; shift 2;;
    -w|--weights)      WEIGHTS=$2; shift 2;;
    -d|--device)       DEV=$2; shift 2;;
    -o|--output)       OUTROOT=$2; shift 2;;
    -l|--log-dir)      LOGDIR=$2; shift 2;;
    -y|--hyp)          HYP=$2; shift 2;;
    -h|--help)         show_help; exit 0;;
    --) shift; EXTRA_ARGS+=("$@"); break;;
    *)  EXTRA_ARGS+=("$1"); shift;;
  esac
done

# Zorunlu alanlar kontrol
need BATCH; need PROJ; need IMG; need EPOCHS

# Varsayılanlar
WEIGHTS=${WEIGHTS:-yolov7/yolov7.pt}
DEV=${DEV:-0}
OUTROOT=${OUTROOT:-trainOutput}
LOGDIR=${LOGDIR:-logs}
HYP=${HYP:-yolov7/data/hyp.scratch.custom.yaml}

# Yol normalleştirme
ROOT=$(pwd)
WEIGHTS=$(real "$WEIGHTS")
HYP=$(real "$HYP")
OUTROOT=$(real "$OUTROOT")
LOGDIR=$(real "$LOGDIR")
PROJ_DIR=$(real "$ROOT/projects/$PROJ")
CFG=$(real "$PROJ_DIR/yolov7.yaml")
DATA=$(real "$PROJ_DIR/data.yaml")

# Dosya kontrol
for f in "$HYP" "$CFG" "$DATA"; do [[ -f $f ]] || { echo "Dosya bulunamadı: $f"; exit 1; }; done
auto_dl "$WEIGHTS"

# Log başlat
mkdir -p "$LOGDIR" "$OUTROOT"
LOG_FILE="$LOGDIR/${PROJ}_$(date +%F_%H-%M-%S).log"
log "=== Eğitim başlatıldı: $PROJ ==="
log "Batch Size   : $BATCH"
log "Image Size   : $IMG"
log "Epochs       : $EPOCHS"
log "Device       : $DEV"
log "Weights      : $WEIGHTS"
log "Hyperparams  : $HYP"
log "Output Dir   : $OUTROOT"
log "Extra Args   : ${EXTRA_ARGS[*]:-(none)}"

# Eğitim başlat
python3 yolov7/train.py \
  --workers 8 \
  --device "$DEV" \
  --batch-size "$BATCH" \
  --img "$IMG" "$IMG" \
  --data "$DATA" \
  --cfg "$CFG" \
  --epochs "$EPOCHS" \
  --project "$OUTROOT" \
  --name "$PROJ" \
  --weights "$WEIGHTS" \
  --hyp "$HYP" \
  "${EXTRA_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"

# Sonuç kontrol
status=${PIPESTATUS[0]}
[[ $status -eq 0 ]] && log "✅ Eğitim tamamlandı." || log "❌ Eğitim başarısız (exit $status)."

# Sesli bildirim (isteğe bağlı, sox gerektirir)
for i in {1..3}; do
  command -v play &>/dev/null && play -nq -t alsa synth 0.2 sine 2000 || break
  sleep 0.2
done

exit $status

