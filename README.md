# FastPlateOCR Training Pipeline — Indian License Plates

A complete end-to-end pipeline for dataset preparation, fine-tuning, and ONNX export of **FastPlateOCR** (Compact Convolutional Transformer - `CCT-S-v2`) for Indian license plates (HSRP, BH series, and DL series).

---

## 📋 Table of Contents
- [Architecture & Overview](#architecture--overview)
- [Directory Structure](#directory-structure)
- [Step 0: Prerequisites & Environment Setup](#step-0-prerequisites--environment-setup)
- [Step 1: Plate Cropping (Tight Crop)](#step-1-plate-cropping-tight-crop)
- [Step 2: Data Preparation Pipeline](#step-2-data-preparation-pipeline)
  - [2.1 Label Extraction](#21-label-extraction-from-filenames)
  - [2.2 Dataset Splitting](#22-dataset-splitting-train--val--test)
  - [2.3 FastPlate Format Conversion](#23-fastplate-dataset-structuring)
  - [2.4 Column Alignment (Critical)](#24-fix-csv-columns-critical)
- [Step 3: Configuration & Pretrained Weights](#step-3-configuration--pretrained-weights)
- [Step 4: Model Training](#step-4-model-training)
- [Step 5: Model Export (ONNX) & Inference](#step-5-model-export-onnx--inference)
- [Troubleshooting & Common Pitfalls](#troubleshooting--common-pitfalls)

---

## 🔍 Architecture & Overview

- **Backbone**: Compact Convolutional Transformer (`CCT-S-v2`) with 5 transformer encoder layers, sequence pooling, and token reduction.
- **Input Dimensions**: 64 × 128 RGB (keep aspect ratio = true, area interpolation).
- **Max Characters / Slots**: 10 slots (`0-9`, `A-Z`, and `_` padding character, 36 unique alphanumeric characters).
- **Supported Plate Standards**:
  - **HSRP**: `^[A-Z]{2}\d{1,2}[A-Z]{1,2}\d{4}$` (e.g. `UP65CF3993`, `MH31FA6976`)
  - **BH Series**: `^\d{2}BH\d{4}[A-Z]{1,2}$` (e.g. `24BH1239C`)
  - **DL (Delhi Series)**: `^DL\d{1,2}[A-Z]{1,3}\d{4}$` (e.g. `DL14CJ2049`)

---

## 📁 Directory Structure

```text
FastplateOCR_Training/
├── README.md                           # Project documentation
├── requirements.txt                    # Project Python dependencies
├── plate_config.yaml                   # License plate format & dimension config
├── cct_s_v2_model_config.yaml          # Model architecture specification
├── fast_plate_ocr_training_guide.txt   # Step-by-step raw training notes
│
├── prepare_snapshots_tight_crop.py     # Step 1: YOLO plate detection & 6% crop
├── extract_labels_from_filenames.py    # Step 2.1: Regex plate label extraction
├── split_dataset.py                    # Step 2.2: 80/10/10 train/val/test split
├── convert_to_fastplate_format.py      # Step 2.3: Build fastplate_data structure
├── fix_csv_columns.py                  # Step 2.4: Align annotation column names
│
├── cct_s_v2_global.keras               # Base pretrained weights (17MB)
├── labels.csv                          # Extracted filename -> plate text mapping
├── failed_extraction.txt               # Log of unparseable image filenames
├── splits/                             # Generated train.csv, val.csv, test.csv
├── fastplate_data/                     # Structured train/val/test splits & images
└── runs/                               # Training checkpoints, logs, and exports
```

---

## ⚙️ Step 0: Prerequisites & Environment Setup

### 1. Create and Activate Virtual Environment (Python 3.12 recommended)
```bash
python3 -m venv venv
source venv/bin/activate       # On Linux/macOS
# On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify Installation
```bash
# Check FastPlateOCR CLI
fast-plate-ocr --help

# Verify PaddleOCR
python3 -c "from paddleocr import PaddleOCR; print('PaddleOCR OK')"

# Verify FastPlateOCR import
python3 -c "from fast_plate_ocr import LicensePlateRecognizer; print('fast-plate-ocr OK')"
```

> [!NOTE]
> **Hardware Compatibility (NVIDIA RTX 50-series / sm_120)**:
> Newer architectures with compute capability 12.0 (sm_120) may not have official upstream CUDA binaries in standard PyTorch or TensorFlow wheels yet. Training default commands use CPU or custom-built sm_120 wheels. To verify device visibility:
> ```bash
> nvidia-smi
> python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
> ```

---

## ✂️ Step 1: Plate Cropping (Tight Crop)

Converts full ANPR snapshot camera images into tight, padded license plate crops using YOLO.

- **Script**: [`prepare_snapshots_tight_crop.py`](file:///home/moodscope/Desktop/Speed_Ocr/FastplateOCR_Training/prepare_snapshots_tight_crop.py)
- **Input**: Full camera snapshots directory (e.g. `training_data/`)
- **Output**: Cropped license plate images (`tight_crop_training_data/`)

```bash
python prepare_snapshots_tight_crop.py
```

**Key Parameters**:
- `PRIMARY_CONF = 0.30`, `FALLBACK_CONF = 0.15`
- `PAD_FRACTION = 0.06` (adds 6% padding around the bounding box to prevent clipping edge characters)
- Minimum crop size guard: `height > 15px`, `width > 30px`

---

## 🔄 Step 2: Data Preparation Pipeline

### 2.1 Label Extraction from Filenames
Extracts plate labels from filenames where the plate number is embedded between underscores (e.g. `lane_4_UP65CF3993_20260217_092439.jpg` → `UP65CF3993`).

- **Script**: [`extract_labels_from_filenames.py`](file:///home/moodscope/Desktop/Speed_Ocr/FastplateOCR_Training/extract_labels_from_filenames.py)
- **Input**: Image crops directory
- **Output**: `labels.csv` (`filename, plate_text`) and `failed_extraction.txt`

```bash
python extract_labels_from_filenames.py
```
> [!TIP]
> If extraction rate is lower than expected, review `failed_extraction.txt` for unusual naming patterns, missing underscores, or multi-line plates.

---

### 2.2 Dataset Splitting (Train / Val / Test)
Shuffles and splits the labeled samples into **80% Train**, **10% Validation**, and **10% Test** splits.

- **Script**: [`split_dataset.py`](file:///home/moodscope/Desktop/Speed_Ocr/FastplateOCR_Training/split_dataset.py)
- **Input**: `labels.csv` (or state-specific label CSVs)
- **Output**: `splits/train.csv`, `splits/val.csv`, `splits/test.csv`

```bash
python split_dataset.py
```

**Dataset Quality Checks**:
- Unique characters must be exactly **36** (`0-9`, `A-Z`). Lowercase letters or symbols indicate malformed labels.
- Format distribution (HSRP vs. BH vs. DL) is logged for validation.

---

### 2.3 FastPlate Dataset Structuring
Copies images into split-specific folders and writes annotations in FastPlateOCR directory format.

- **Script**: [`convert_to_fastplate_format.py`](file:///home/moodscope/Desktop/Speed_Ocr/FastplateOCR_Training/convert_to_fastplate_format.py)
- **Input**: `splits/*.csv` + source images
- **Output**: `fastplate_data/{train,val,test}/{images/, annotations.csv}`

```bash
python convert_to_fastplate_format.py
```

---

### 2.4 Fix CSV Columns (⚠️ CRITICAL)
Ensures exact column headers and relative image paths required by `fast-plate-ocr` v1.1.0.

- **Script**: [`fix_csv_columns.py`](file:///home/moodscope/Desktop/Speed_Ocr/FastplateOCR_Training/fix_csv_columns.py)
- **Input / Output**: `fastplate_data/{split}/annotations.csv` (modified in place)

```bash
python fix_csv_columns.py
```

**Verification**:
```bash
head -n 3 fastplate_data/train/annotations.csv
```
Must display:
```csv
image_path,plate_text
images/lane_4_UP65CF3993_20260217_092439.jpg,UP65CF3993
images/lane_1_BR24GD2995_20260108_154911.jpg,BR24GD2995
```

---

## 🛠️ Step 3: Configuration & Pretrained Weights

### 1. Plate Config ([`plate_config.yaml`](file:///home/moodscope/Desktop/Speed_Ocr/FastplateOCR_Training/plate_config.yaml))
Defines dimensions and alphabet rules:
```yaml
max_plate_slots: 10
alphabet: '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_'
pad_char: '_'
img_height: 64
img_width: 128
keep_aspect_ratio: true
interpolation: area
image_color_mode: rgb
```

### 2. Model Config ([`cct_s_v2_model_config.yaml`](file:///home/moodscope/Desktop/Speed_Ocr/FastplateOCR_Training/cct_s_v2_model_config.yaml))
Specifies the CCT-S-v2 transformer architecture. If missing, download it via:
```bash
curl -L "https://github.com/ankandrew/fast-plate-ocr/releases/download/arg-plates/cct_s_v2_global_model_config.yaml" -o cct_s_v2_model_config.yaml
```

### 3. Pretrained Weights (`cct_s_v2_global.keras`)
Download the pretrained global weights using the Python API:
```bash
python3 -c "from fast_plate_ocr import LicensePlateRecognizer; m = LicensePlateRecognizer('cct-s-v2-global-model'); print('Downloaded.')"
```
Locate the downloaded `.keras` file and copy it to the project root:
```bash
find ~/.cache -name "*cct*.keras" 2>/dev/null
# or find / -name "*.keras" 2>/dev/null
cp <path_to_cached_model>.keras cct_s_v2_global.keras
```
Verify the file is a valid Keras zip archive:
```bash
file cct_s_v2_global.keras
# Should output: Zip archive data
```

---

## 🚀 Step 4: Model Training

Execute the training command using the PyTorch Keras backend.

### Recommended: Run inside `tmux`
To prevent interruptions during long training sessions over SSH:
```bash
tmux new -s fastplate
```
*(Detach with `Ctrl+B` then `D`; reattach anytime with `tmux attach -t fastplate`)*

### Training Command
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
KERAS_BACKEND=torch \
fast-plate-ocr train \
  --model-config-file cct_s_v2_model_config.yaml \
  --plate-config-file plate_config.yaml \
  --annotations fastplate_data/train/annotations.csv \
  --val-annotations fastplate_data/val/annotations.csv \
  --weights-path cct_s_v2_global.keras \
  --batch-size 16 \
  --epochs 150 \
  --lr 5e-4 \
  --use-ema \
  --early-stopping-patience 40 \
  --early-stopping-metric val_plate_acc \
  --workers 1 \
  --output-dir runs/indian_plates_v1_tight
```

### Monitoring Training
- Training logs are saved to `runs/indian_plates_v1_tight/<timestamp>/training_log.csv`.
- Best checkpoints are saved to `runs/indian_plates_v1_tight/<timestamp>/best.keras`.
- Target: `val_plate_acc > 90%` typically by epoch 30–50.

---

## 📦 Step 5: Model Export (ONNX) & Inference

### 1. Export Best Checkpoint to ONNX
Exporting converts the Keras model to a lightweight, standalone ONNX runtime format for production inference:
```bash
fast-plate-ocr export \
  --model-path runs/indian_plates_v1_tight/<timestamp>/best.keras \
  --output-format onnx \
  --output-path runs/indian_plates_v1_tight/best.onnx
```

### 2. Verify Inference in Python
```python
from fast_plate_ocr import LicensePlateRecognizer

# Load fine-tuned ONNX model
recognizer = LicensePlateRecognizer(
    model_path="runs/indian_plates_v1_tight/best.onnx"
)

# Run prediction on a test crop
result = recognizer.run("fastplate_data/test/images/sample.jpg")
print(f"Recognized Plate: {result}")
```

---

## ⚠️ Troubleshooting & Common Pitfalls

| Issue / Error | Cause | Solution |
| :--- | :--- | :--- |
| `KeyError: 'image_path'` | Annotation CSV has `image` or `filename` header | Run [`fix_csv_columns.py`](file:///home/moodscope/Desktop/Speed_Ocr/FastplateOCR_Training/fix_csv_columns.py) to rename columns |
| `KeyError: 'plate_text'` | Annotation CSV has `label` or `text` header | Run [`fix_csv_columns.py`](file:///home/moodscope/Desktop/Speed_Ocr/FastplateOCR_Training/fix_csv_columns.py) to rename columns |
| `FileNotFoundError` (doubled paths) | Paths in CSV are absolute instead of relative | Make paths relative: `images/<filename>.jpg` |
| Low accuracy on BH plates | Underrepresented in dataset (<100 samples) | Collect additional BH plate samples or apply data augmentation |
| Character count ≠ 36 | Lowercase letters or symbols in `plate_text` | Filter labels in [`split_dataset.py`](file:///home/moodscope/Desktop/Speed_Ocr/FastplateOCR_Training/split_dataset.py) |
| Model dimension mismatch | Changed `img_height` or `img_width` in config | Dimensions must remain **64x128** when loading `cct_s_v2_global.keras` |
