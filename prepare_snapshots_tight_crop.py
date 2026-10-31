import os
from pathlib import Path

import cv2
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_DIR = "training_data"
OUTPUT_DIR = "tight_crop_training_data"

LP_MODEL = (
    "/home/moodscope/Desktop/Speed_Ocr/"
    "speed_ocr_testing/highway-detection/models/license_plate.pt"
)

PRIMARY_CONF = 0.30
FALLBACK_CONF = 0.15
PAD_FRACTION = 0.06

DEVICE = "0"
IMGSZ = 320


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# CROP PLATE FROM YOLO RESULT
# ============================================================

def crop_from_result(img, result):
    """
    Take the highest-confidence YOLO plate detection,
    add 6% padding and return the cropped plate.
    """

    if result is None or result.boxes is None or len(result.boxes) == 0:
        return None

    # Highest-confidence box
    lp_box = result.boxes[0]

    lx1, ly1, lx2, ly2 = map(
        int,
        lp_box.xyxy[0].tolist()
    )

    h, w = img.shape[:2]

    # 6% padding
    pad_w = int((lx2 - lx1) * PAD_FRACTION)
    pad_h = int((ly2 - ly1) * PAD_FRACTION)

    lx1 = max(0, lx1 - pad_w)
    ly1 = max(0, ly1 - pad_h)

    lx2 = min(w, lx2 + pad_w)
    ly2 = min(h, ly2 + pad_h)

    crop = img[ly1:ly2, lx1:lx2]

    # Minimum crop size
    if (
        crop.size > 0
        and crop.shape[0] > 15
        and crop.shape[1] > 30
    ):
        return crop

    return None


# ============================================================
# MAIN
# ============================================================

def main():

    image_dir = Path(IMAGE_DIR)
    output_dir = Path(OUTPUT_DIR)

    if not image_dir.exists():
        print(f"[ERROR] Image directory does not exist: {image_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # FIND IMAGES
    # --------------------------------------------------------

    files = sorted(
        f for f in image_dir.iterdir()
        if f.is_file()
        and f.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    print("=" * 60)
    print(f"Input directory  : {image_dir}")
    print(f"Output directory : {output_dir}")
    print(f"Images found     : {len(files)}")
    print(f"Model            : {LP_MODEL}")
    print(f"Primary conf     : {PRIMARY_CONF}")
    print(f"Fallback conf    : {FALLBACK_CONF}")
    print(f"Padding          : {PAD_FRACTION * 100:.0f}%")
    print(f"Device           : {DEVICE}")
    print(f"Image size       : {IMGSZ}")
    print("=" * 60)

    # --------------------------------------------------------
    # LOAD YOLO MODEL
    # --------------------------------------------------------

    print("[INFO] Loading YOLO model...")

    model = YOLO(LP_MODEL)

    print("[INFO] Model loaded")

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    total = 0
    primary_count = 0
    fallback_count = 0
    missed_count = 0
    read_failed_count = 0

    # --------------------------------------------------------
    # PROCESS IMAGES
    # --------------------------------------------------------

    for index, image_path in enumerate(files, start=1):

        total += 1

        print(
            f"[{index}/{len(files)}] "
            f"{image_path.name}",
            end=" "
        )

        # ----------------------------------------------------
        # READ IMAGE
        # ----------------------------------------------------

        img = cv2.imread(str(image_path))

        if img is None:
            print("[READ FAILED]")

            read_failed_count += 1
            continue

        # ----------------------------------------------------
        # PRIMARY DETECTION
        # ----------------------------------------------------

        results = model(
            img,
            conf=PRIMARY_CONF,
            verbose=False,
            device=DEVICE,
            imgsz=IMGSZ
        )

        crop = crop_from_result(img, results[0])

        detection_type = "primary"

        # ----------------------------------------------------
        # FALLBACK DETECTION
        # ----------------------------------------------------

        if crop is None:

            results = model(
                img,
                conf=FALLBACK_CONF,
                verbose=False,
                device=DEVICE,
                imgsz=IMGSZ
            )

            crop = crop_from_result(img, results[0])

            detection_type = "fallback"

        # ----------------------------------------------------
        # NO DETECTION
        # ----------------------------------------------------

        if crop is None:

            print("[NOT DETECTED]")

            missed_count += 1
            continue

        # ----------------------------------------------------
        # SAVE CROP
        # ----------------------------------------------------

        output_path = output_dir / image_path.name

        success = cv2.imwrite(
            str(output_path),
            crop
        )

        if not success:
            print("[SAVE FAILED]")
            continue

        # ----------------------------------------------------
        # UPDATE STATS
        # ----------------------------------------------------

        if detection_type == "primary":
            primary_count += 1
        else:
            fallback_count += 1

        h, w = crop.shape[:2]

        print(
            f"[{detection_type}] "
            f"crop={w}x{h}"
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)

    print(f"Total images      : {total}")
    print(f"Primary detected  : {primary_count}")
    print(f"Fallback detected : {fallback_count}")
    print(f"Not detected      : {missed_count}")
    print(f"Read failed       : {read_failed_count}")
    print(
        f"Successfully crop : "
        f"{primary_count + fallback_count}"
    )

    print()
    print(f"Output directory  : {output_dir}")
    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
