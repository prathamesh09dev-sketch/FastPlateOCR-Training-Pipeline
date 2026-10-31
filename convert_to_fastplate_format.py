import csv
import os
import shutil

IMAGE_DIR = 'all_types_data'
SPLITS_DIR = 'splits'
OUTPUT_DIR = 'fastplate_data'

os.makedirs(OUTPUT_DIR, exist_ok=True)

for split in ['train', 'val', 'test']:
    # Create folder for this split's images
    split_img_dir = os.path.join(OUTPUT_DIR, split, 'images')
    os.makedirs(split_img_dir, exist_ok=True)
    
    # Read split CSV and create annotation file
    annotations = []
    with open(os.path.join(SPLITS_DIR, f'{split}.csv')) as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = os.path.join(IMAGE_DIR, row['filename'])
            dst = os.path.join(split_img_dir, row['filename'])
            if os.path.exists(src):
                # Copy (or symlink to save space) image to split folder
                if not os.path.exists(dst):
                    shutil.copy(src, dst)
                annotations.append([row['filename'], row['plate_text']])
    
    # Write annotation CSV in fast-plate-ocr format
    ann_path = os.path.join(OUTPUT_DIR, split, 'annotations.csv')
    with open(ann_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'plate_text'])
        writer.writerows(annotations)
    
    print(f'{split}: {len(annotations)} samples → {ann_path}')

print('\nDone. Data structured for fast-plate-ocr training.')
