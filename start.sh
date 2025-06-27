#!/bin/bash

# Constants
BASE_DIR=$(pwd)

# Help message
function display_help() {
    echo "Usage: ./start.sh [OPTIONS]"
    echo
    echo "Options:"
    echo "  -p, --project     Project name"
    echo "  -r, --ratio       Train-validation split ratio (e.g., 0.9)"
    echo "  -c, --classes     Class names (space separated)"
    echo "  -h, --help        Show this help message"
    echo
    echo "Example:"
    echo "./start.sh -p Autoloader -r 0.9 -c \"CHR_FR CHR_R COR_FR COR_RR\""
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            display_help
            exit 0
            ;;
        -p|--project)
            PROJECT_NAME="$2"
            shift 2
            ;;
        -r|--ratio)
            RATIO="$2"
            shift 2
            ;;
        -c|--classes)
            CLASS_NAMES_STR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            display_help
            exit 1
            ;;
    esac
done

# Prompt if anything missing
if [[ -z "$PROJECT_NAME" ]]; then
    read -p "Enter project name: " PROJECT_NAME
fi
if [[ -z "$RATIO" ]]; then
    read -p "Enter train/val ratio (e.g. 0.9): " RATIO
fi
if [[ -z "$CLASS_NAMES_STR" ]]; then
    read -p "Enter class names (space-separated): " CLASS_NAMES_STR
fi

# Activate environment
source yololab7/bin/activate

# Run project setup with your custom script
echo "🚀 Proje kuruluyor..."
python3 train_valid_split.py "$PROJECT_NAME" "$RATIO" $CLASS_NAMES_STR

echo ""
echo "✅ Proje '$PROJECT_NAME' başarıyla oluşturuldu!"

