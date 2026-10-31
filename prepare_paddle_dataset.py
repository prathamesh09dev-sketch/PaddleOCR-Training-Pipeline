# PaddleOCR Dataset Conversion Script

#!/usr/bin/env python3

import argparse
import csv
import random
import shutil
from pathlib import Path


def find_column(fieldnames, candidates):
    """Find a CSV column using case-insensitive matching."""
    normalized = {
        field.strip().lower(): field
        for field in fieldnames
    }

    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]

    return None


def clean_label(label):
    """Clean license plate text."""
    if label is None:
        return ""

    # Remove spaces and convert to uppercase
    label = str(label).strip().upper()
    label = label.replace(" ", "")
    label = label.replace("-", "")

    return label


def main():

    parser = argparse.ArgumentParser(
        description="Convert license plate images + labels.csv into PaddleOCR dataset"
    )

    parser.add_argument(
        "--images",
        required=True,
        help="Path to image folder"
    )

    parser.add_argument(
        "--csv",
        required=True,
        help="Path to labels.csv"
    )

    parser.add_argument(
        "--output",
        default="paddle_plate_dataset",
        help="Output dataset directory"
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.20,
        help="Validation ratio. Default: 0.20"
    )

    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.10,
        help="Test ratio. Default: 0.10"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    image_dir = Path(args.images)
    csv_path = Path(args.csv)
    output_dir = Path(args.output)

    if not image_dir.exists():
        raise FileNotFoundError(
            f"Image folder does not exist: {image_dir}"
        )

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file does not exist: {csv_path}"
        )

    # ---------------------------------------------------------
    # Create output directories
    # ---------------------------------------------------------

    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    test_dir = output_dir / "test"
    labels_dir = output_dir / "labels"

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Read CSV
    # ---------------------------------------------------------

    print(f"[INFO] Reading CSV: {csv_path}")

    with open(
        csv_path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        if not reader.fieldnames:
            raise ValueError("CSV has no header.")

        print("[INFO] CSV columns:")
        for column in reader.fieldnames:
            print(f"       {column}")

        # Try common filename columns
        filename_col = find_column(
            reader.fieldnames,
            [
                "filename",
                "file_name",
                "image",
                "image_name",
                "image_filename",
                "path",
            ]
        )

        # Try common license plate columns
        label_col = find_column(
            reader.fieldnames,
            [
                "license_plate_text",
                "license_plate",
                "plate_text",
                "plate",
                "label",
                "text",
                "ocr",
            ]
        )

        if filename_col is None:
            raise ValueError(
                "Could not find image filename column in CSV.\n"
                "Expected something like: filename / image / image_name"
            )

        if label_col is None:
            raise ValueError(
                "Could not find license plate label column in CSV.\n"
                "Expected something like: license_plate_text / plate / label"
            )

        print(f"[INFO] Filename column : {filename_col}")
        print(f"[INFO] Label column    : {label_col}")

        records = []

        for row in reader:

            filename = str(row.get(filename_col, "")).strip()
            label = clean_label(row.get(label_col, ""))

            if not filename:
                continue

            if not label:
                continue

            records.append(
                {
                    "filename": filename,
                    "label": label,
                }
            )

    print(f"[INFO] CSV records with labels: {len(records)}")

    # ---------------------------------------------------------
    # Find images
    # ---------------------------------------------------------

    valid_records = []
    missing_images = []

    for record in records:

        filename = record["filename"]

        # First try exact path
        image_path = image_dir / filename

        if not image_path.exists():

            # Try only basename
            image_path = image_dir / Path(filename).name

        if not image_path.exists():
            missing_images.append(filename)
            continue

        record["image_path"] = image_path

        valid_records.append(record)

    print(
        f"[INFO] Images found : {len(valid_records)}"
    )

    print(
        f"[INFO] Images missing: {len(missing_images)}"
    )

    # ---------------------------------------------------------
    # Remove duplicate filenames
    # ---------------------------------------------------------

    unique = {}

    for record in valid_records:

        filename = record["filename"]

        if filename not in unique:
            unique[filename] = record

    valid_records = list(unique.values())

    print(
        f"[INFO] Unique images: {len(valid_records)}"
    )

    if len(valid_records) == 0:
        raise RuntimeError(
            "No valid images found. Check your CSV filename column "
            "and image folder."
        )

    # ---------------------------------------------------------
    # Shuffle
    # ---------------------------------------------------------

    random.seed(args.seed)

    random.shuffle(valid_records)

    # ---------------------------------------------------------
    # Train / validation split
    # ---------------------------------------------------------

    # Calculate validation and test sizes
    val_count = int(len(valid_records) * args.val_ratio)
    test_count = int(len(valid_records) * args.test_ratio)

    # Make sure validation/test are not empty when enough images exist
    if len(valid_records) >= 3:
        val_count = max(1, val_count)
        test_count = max(1, test_count)

    # Prevent train from becoming empty
    if val_count + test_count >= len(valid_records):
        test_count = max(1, len(valid_records) // 10)
        val_count = max(1, len(valid_records) // 5)

    train_end = len(valid_records) - val_count - test_count
    val_end = len(valid_records) - test_count

    train_records = valid_records[:train_end]
    val_records = valid_records[train_end:val_end]
    test_records = valid_records[val_end:]

    print()
    print("Dataset split")
    print("-------------------------")
    print(f"Total : {len(valid_records)}")
    print(f"Train : {len(train_records)}")
    print(f"Val   : {len(val_records)}")
    print(f"Test  : {len(test_records)}")
    print()

    # ---------------------------------------------------------
    # Copy images and create labels
    # ---------------------------------------------------------

    train_annotations = []
    val_annotations = []
    test_annotations = []

    def process_records(records, destination, annotations):

        for index, record in enumerate(records, start=1):

            source = record["image_path"]
            label = record["label"]

            # -------------------------------------------------
            # Create a clean filename
            #
            # Example:
            #
            # MH12AB1234_00001.jpg
            # -------------------------------------------------

            extension = source.suffix.lower()

            if extension not in [
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".webp",
            ]:
                extension = ".jpg"

            new_filename = (
                f"{label}_{index:05d}{extension}"
            )

            destination_path = (
                destination / new_filename
            )

            shutil.copy2(
                source,
                destination_path
            )

            # PaddleOCR annotation format:
            #
            # relative/path/to/image.jpg<TAB>TEXT
            #
            annotations.append(
                f"{destination.name}/{new_filename}\t{label}"
            )

            if index % 500 == 0:
                print(
                    f"       processed {index}/{len(records)}"
                )

    print("[INFO] Processing training images...")

    process_records(
        train_records,
        train_dir,
        train_annotations
    )

    print("[INFO] Processing validation images...")

    process_records(
        val_records,
        val_dir,
        val_annotations
    )

    print("[INFO] Processing test images...")

    process_records(
        test_records,
        test_dir,
        test_annotations
    )

    # ---------------------------------------------------------
    # Write train.txt
    # ---------------------------------------------------------

    train_txt = labels_dir / "train.txt"

    with open(
        train_txt,
        "w",
        encoding="utf-8"
    ) as f:

        for line in train_annotations:
            f.write(line + "\n")

    # ---------------------------------------------------------
    # Write val.txt
    # ---------------------------------------------------------

    val_txt = labels_dir / "val.txt"

    with open(
        val_txt,
        "w",
        encoding="utf-8"
    ) as f:

        for line in val_annotations:
            f.write(line + "\n")

    # ---------------------------------------------------------
    # Write test.txt
    # ---------------------------------------------------------

    test_txt = labels_dir / "test.txt"

    with open(
        test_txt,
        "w",
        encoding="utf-8"
    ) as f:

        for line in test_annotations:
            f.write(line + "\n")

    # ---------------------------------------------------------
    # Print summary
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("PADDLEOCR DATASET CREATED")
    print("=" * 60)

    print(f"Output directory : {output_dir}")
    print(f"Train images     : {len(train_records)}")
    print(f"Validation images: {len(val_records)}")
    print(f"Test images      : {len(test_records)}")
    print(f"Train labels     : {train_txt}")
    print(f"Val labels       : {val_txt}")
    print(f"Test labels      : {test_txt}")

    print()
    print("Example train annotation:")

    if train_annotations:
        print(train_annotations[0])

    print()
    print("Example validation annotation:")

    if val_annotations:
        print(val_annotations[0])

    print()
    print("Example test annotation:")

    if test_annotations:
        print(test_annotations[0])

    # ---------------------------------------------------------
    # Missing image report
    # ---------------------------------------------------------

    if missing_images:

        missing_file = output_dir / "missing_images.txt"

        with open(
            missing_file,
            "w",
            encoding="utf-8"
        ) as f:

            for filename in missing_images:
                f.write(filename + "\n")

        print()
        print(
            f"[WARNING] Missing image list saved to: "
            f"{missing_file}"
        )


if __name__ == "__main__":
    main()