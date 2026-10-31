import os
import csv
import re

IMAGE_DIR = '/home/moodscope/Desktop/Speed_Ocr/ocr_training/Snapshots_tight_crop/'        # ← change to your folder
OUTPUT_CSV = 'labels.csv'

# Pattern to extract plate text from filename
# Format observed: lane_4_UP65CF3993_20260217_092439.jpg
# Plate is the 3rd underscore-separated chunk
HSRP = re.compile(r'^[A-Z]{2}\d{1,2}[A-Z]{1,2}\d{4}$')
BH = re.compile(r'^\d{2}BH\d{4}[A-Z]{1,2}$')

files = [f for f in os.listdir(IMAGE_DIR)
         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
print(f'Processing {len(files)} images...')

extracted = 0
failed = []
invalid_format = []

with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['filename', 'plate_text'])
    
    for fname in files:
        # Split by underscore, find the chunk that matches a plate pattern
        parts = fname.replace('.jpg', '').replace('.jpeg', '').replace('.png', '').split('_')
        
        plate = None
        for part in parts:
            part_upper = part.upper()
            if HSRP.match(part_upper) or BH.match(part_upper):
                plate = part_upper
                break
        
        if plate:
            writer.writerow([fname, plate])
            extracted += 1
        else:
            failed.append(fname)

print(f'\nExtracted: {extracted}/{len(files)} ({extracted/len(files)*100:.1f}%)')
print(f'Failed:    {len(failed)}')

if failed:
    print(f'\nFirst 10 failures (filename pattern may differ):')
    for f in failed[:10]:
        print(f'  {f}')
    
    # Save failures for inspection
    with open('failed_extraction.txt', 'w') as f:
        f.write('\n'.join(failed))
    print(f'\nAll {len(failed)} failures saved to failed_extraction.txt')
