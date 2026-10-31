# save as: fix_csv_columns.py
import pandas as pd
import os

for split in ['train', 'val', 'test']:
    path = f'fastplate_data/{split}/annotations.csv'
    df = pd.read_csv(path)

    # Rename columns to what fast-plate-ocr expects
    col_map = {}
    for col in df.columns:
        if col in ['image', 'filename']:
            col_map[col] = 'image_path'
        if col in ['label', 'text']:
            col_map[col] = 'plate_text'
    df = df.rename(columns=col_map)

    # CRITICAL: paths must be relative to CSV location, with images/ prefix
    df['image_path'] = df['image_path'].apply(
        lambda f: f'images/{os.path.basename(f)}' if not f.startswith('images/') else f
    )

    df.to_csv(path, index=False)
    print(f'{split} fixed: columns={list(df.columns)}')
    print(f'  Sample: {df.iloc[0].to_dict()}')
    print()

print('All CSVs fixed.')
