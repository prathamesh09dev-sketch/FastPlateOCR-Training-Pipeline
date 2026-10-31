# import csv
# import random
# import os
# from collections import Counter

# LABELS_CSV = 'labels.csv'
# OUTPUT_DIR = 'splits'
# SEED = 42

# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # Load labels
# with open(LABELS_CSV) as f:
#     reader = csv.reader(f)
#     next(reader)  # skip header
#     rows = [row for row in reader if row[1]]  # skip empty labels

# print(f'Total labeled samples: {len(rows)}')

# # Check format distribution (BH vs HSRP)
# import re
# HSRP = re.compile(r'^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$')
# BH = re.compile(r'^\d{2}BH\d{4}[A-Z]{1,2}$')

# bh_count = sum(1 for r in rows if BH.match(r[1]))
# hsrp_count = sum(1 for r in rows if HSRP.match(r[1]))
# print(f'  HSRP plates: {hsrp_count}')
# print(f'  BH plates:   {bh_count}')

# # Check character distribution
# all_chars = ''.join(r[1] for r in rows)
# char_counts = Counter(all_chars)
# print(f'\nUnique characters in dataset: {len(char_counts)}')
# print(f'Characters: {sorted(char_counts.keys())}')

# # Shuffle and split 80/10/10
# random.seed(SEED)
# random.shuffle(rows)

# n = len(rows)
# train_end = int(0.8 * n)
# val_end = int(0.9 * n)

# train = rows[:train_end]
# val = rows[train_end:val_end]
# test = rows[val_end:]

# # Write splits
# for name, data in [('train', train), ('val', val), ('test', test)]:
#     out_path = os.path.join(OUTPUT_DIR, f'{name}.csv')
#     with open(out_path, 'w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(['filename', 'plate_text'])
#         writer.writerows(data)
#     print(f'{name}: {len(data)} samples → {out_path}')

# print('\nDone. Splits ready for fine-tuning.')


import csv
import random
import os
import re
from collections import Counter

LABELS_CSV = 'vahan_labels.csv'
OUTPUT_DIR = 'vahan_splits'
SEED = 42
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load labels
with open(LABELS_CSV) as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    rows = [row for row in reader if row[1]]  # skip empty labels

print(f'Total labeled samples: {len(rows)}')

# --- Format patterns (kept in sync with plate_validation.py) ---
HSRP = re.compile(r'^[A-Z]{2}\d{1,2}[A-Z]{1,2}\d{4}$')  # 1-2 digit district
BH = re.compile(r'^\d{2}BH\d{4}[A-Z]{1,2}$')
DL = re.compile(r'^DL\d{1,2}[A-Z]{1,3}\d{4}$')          # up to 3-letter series

hsrp_count = sum(1 for r in rows if HSRP.match(r[1]))
bh_count = sum(1 for r in rows if BH.match(r[1]))
dl_count = sum(1 for r in rows if r[1][:2] == 'DL' and DL.match(r[1]))
unmatched = [r for r in rows if not (
    BH.match(r[1]) or
    (r[1][:2] == 'DL' and DL.match(r[1])) or
    HSRP.match(r[1])
)]

print(f'  HSRP plates: {hsrp_count}')
print(f'  DL plates:   {dl_count}')
print(f'  BH plates:   {bh_count}')
print(f'  Unmatched (bad label format): {len(unmatched)}')
if unmatched:
    print(f'    e.g. {[r[1] for r in unmatched[:10]]}')

# Check character distribution
all_chars = ''.join(r[1] for r in rows)
char_counts = Counter(all_chars)
print(f'\nUnique characters in dataset: {len(char_counts)}')
print(f'Characters: {sorted(char_counts.keys())}')

# Shuffle and split 80/10/10
random.seed(SEED)
random.shuffle(rows)
n = len(rows)
train_end = int(0.8 * n)
val_end = int(0.9 * n)
train = rows[:train_end]
val = rows[train_end:val_end]
test = rows[val_end:]

# Write splits
for name, data in [('train', train), ('val', val), ('test', test)]:
    out_path = os.path.join(OUTPUT_DIR, f'{name}.csv')
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'plate_text'])
        writer.writerows(data)
    print(f'{name}: {len(data)} samples \u2192 {out_path}')

print('\nDone. Splits ready for fine-tuning.')