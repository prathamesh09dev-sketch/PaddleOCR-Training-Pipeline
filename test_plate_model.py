# test_paddle_ocr.py
import argparse
import csv
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path

import cv2
import yaml
from paddleocr import PaddleOCR


# ============================================================
# CONFIG
# ============================================================

MODEL_DIR = "/home/moodscope/Desktop/Speed_Ocr/PaddleOCR_Training/output/plate_rec_v1/inference"
DICT_PATH = "/home/moodscope/Desktop/Speed_Ocr/PaddleOCR_Training/plate_dict.txt"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

VALID_STATES = {
    "AP", "AR", "AS", "BR", "CG", "GA", "GJ", "HR", "HP", "JH",
    "KA", "KL", "MP", "MH", "MN", "ML", "MZ", "NL", "OD", "PB",
    "RJ", "SK", "TN", "TS", "TR", "UP", "UK", "WB",
    "AN", "CH", "DD", "DL", "JK", "LA", "LD", "PY",
}

HSRP_PATTERN = re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{1,2}\d{4}$")
BH_PATTERN = re.compile(r"^\d{2}BH\d{4}[A-Z]{1,2}$")

STATE_CONFUSION = {
    "N": "M", "M": "N", "0": "O", "O": "0",
    "1": "I", "I": "1", "5": "S", "S": "5",
    "8": "B", "B": "8", "D": "O", "Q": "O",
    "U": "V", "V": "U",
}


# ============================================================
# PLATE CLEANING / VALIDATION
# (ported from plate_ocr_pipeline.py)
# ============================================================

def clean_plate_text(text):
    if not text:
        return ""
    text = str(text).upper().strip()
    text = re.sub(r"[\s.\-_]", "", text)
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text


def fix_state_code(state_code):
    if len(state_code) != 2 or state_code in VALID_STATES:
        return state_code
    a, b = state_code[0], state_code[1]
    for candidate in [
        STATE_CONFUSION.get(a, a) + b,
        a + STATE_CONFUSION.get(b, b),
        STATE_CONFUSION.get(a, a) + STATE_CONFUSION.get(b, b),
    ]:
        if candidate in VALID_STATES:
            return candidate
    return state_code


def _is_valid(plate):
    if not plate:
        return False
    if BH_PATTERN.match(plate):
        return True
    return bool(HSRP_PATTERN.match(plate) and plate[:2] in VALID_STATES)


def load_reference_plates(path):
    """Load known plates from .txt (1/line) or .csv (col 0)."""
    if not path or not os.path.exists(path):
        return set()
    plates = set()
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if row:
                    c = clean_plate_text(row[0])
                    if c:
                        plates.add(c)
    else:
        with open(path, encoding="utf-8") as f:
            for line in f:
                c = clean_plate_text(line)
                if c:
                    plates.add(c)
    return plates


def _edit_distance_1_match(plate, reference_set):
    for ref in reference_set:
        if len(ref) != len(plate):
            continue
        if sum(1 for a, b in zip(plate, ref) if a != b) == 1:
            return ref
    return None


def validate_plate(text, reference_set=None):
    """
    clean -> accept-if-already-valid -> reference exact match ->
    state-code fix -> reference fuzzy match.

    Returns (is_valid, plate, was_corrected, source)
    source in: ocr, state_fix, reference_exact, reference_fuzzy, none
    """
    original = clean_plate_text(text)
    if not original or len(original) < 6:
        return False, original, False, "none"

    if _is_valid(original):
        return True, original, False, "ocr"

    if reference_set and original in reference_set:
        return True, original, False, "reference_exact"

    if not original[:2].isdigit():
        fixed_state = fix_state_code(original[:2])
        if fixed_state != original[:2]:
            candidate = fixed_state + original[2:]
            if _is_valid(candidate):
                return True, candidate, True, "state_fix"

    if reference_set:
        fuzzy = _edit_distance_1_match(original, reference_set)
        if fuzzy is not None:
            return True, fuzzy, True, "reference_fuzzy"

    return False, original, False, "none"


# ============================================================
# SYNC CHARACTER DICT INTO inference.yml
# ============================================================

def ensure_character_dict(model_dir, dict_path):
    """
    PaddleOCR 3.x reads the recognition character list from
    PostProcess.character_dict inside <model_dir>/inference.yml -
    there is no runtime `dict_path` kwarg. This patches that key
    from plate_dict.txt (one character per line) if it's missing
    or out of sync, so the trained dict is actually used.
    """
    yml_path = os.path.join(model_dir, "inference.yml")
    if not os.path.exists(yml_path):
        print(f"[WARN] inference.yml not found at {yml_path}; skipping dict sync")
        return
    if not os.path.exists(dict_path):
        print(f"[WARN] dict file not found at {dict_path}; skipping dict sync")
        return

    with open(dict_path, encoding="utf-8") as f:
        chars = [line.rstrip("\n") for line in f if line.rstrip("\n") != ""]

    with open(yml_path, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    post = config.setdefault("PostProcess", {})
    if post.get("character_dict") == chars:
        return  # already in sync

    post["character_dict"] = chars
    post.setdefault("name", "CTCLabelDecode")

    with open(yml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)

    print(f"[INFO] Synced {len(chars)} characters from {dict_path} into {yml_path}")


# ============================================================
# OCR
# ============================================================

def create_ocr():
    """
    Load the trained PaddleOCR recognition model.
    """

    ensure_character_dict(MODEL_DIR, DICT_PATH)

    ocr = PaddleOCR(
        text_recognition_model_dir=MODEL_DIR,
        text_recognition_model_name="PP-OCRv5_mobile_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    return ocr


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_image(ocr, image_path, reference_set=None):
    """
    Run OCR on a single image and return a list of per-detection
    result rows: (image_path, raw_text, score, plate, valid, corrected, source)

    A single image can yield zero, one, or several rows - one row
    per detected text region. Callers that want a single row per
    image should pick_best_detection() over the returned list.
    """
    print("\n" + "=" * 70)
    print(f"Image: {image_path}")
    print("=" * 70)

    rows = []

    try:
        result = ocr.predict(str(image_path))

        found_text = False

        for res in result:

            # PaddleOCR 3.x result objects (OCRResult) are dict
            # subclasses themselves - use directly. .json wraps
            # fields under a "res" key so only fall back to it
            # if res isn't already dict-like.
            if isinstance(res, dict):
                data = res
            elif hasattr(res, "json"):
                data = res.json
                if callable(data):
                    data = data()
                if isinstance(data, str):
                    data = json.loads(data)
                if isinstance(data, dict) and "res" in data:
                    data = data["res"]
            else:
                data = {}

            # ------------------------------------------------
            # Try to extract recognition result
            # ------------------------------------------------

            rec_texts = data.get("rec_texts", [])
            rec_scores = data.get("rec_scores", [])

            if rec_texts:
                found_text = True

                for i, text in enumerate(rec_texts):

                    score = (
                        rec_scores[i]
                        if i < len(rec_scores)
                        else 0.0
                    )

                    is_valid, plate, corrected, source = validate_plate(
                        text, reference_set=reference_set
                    )
                    status = "VALID" if is_valid else "NOT MATCH"

                    print(f"Text     : {text}")
                    print(f"Score    : {score:.4f}")
                    print(f"Plate    : {plate}")
                    print(f"Status   : {status}")
                    print(f"Corrected: {corrected}  (source={source})")

                    rows.append(
                        (
                            str(image_path),
                            text,
                            f"{score:.4f}",
                            plate,
                            is_valid,
                            corrected,
                            source,
                        )
                    )

        if not found_text:
            print("Text  : <NO TEXT>")
            rows.append((str(image_path), "", "", "", False, False, "none"))

    except Exception as e:
        print(f"[ERROR] {e}")
        rows.append((str(image_path), "", "", "", False, False, "error"))

    return rows


def pick_best_detection(rows):
    """
    Collapse all detections for ONE image down to a single
    representative row, so counting is done per-image instead of
    per-detected-text-region.

    Preference order:
      1. A valid detection, highest score wins among valid ones.
      2. If none are valid, the highest-scoring detection overall.
      3. If there are no rows at all (shouldn't happen - process_image
         always returns at least a NO TEXT row), return a NO TEXT row.
    """
    if not rows:
        return (None, "", "", "", False, False, "none")

    def score_of(row):
        try:
            return float(row[2]) if row[2] != "" else -1.0
        except ValueError:
            return -1.0

    valid_rows = [r for r in rows if r[4] is True]
    if valid_rows:
        return max(valid_rows, key=score_of)

    return max(rows, key=score_of)


# ============================================================
# GET IMAGES
# ============================================================

def get_images(input_path):

    path = Path(input_path)

    # Single image
    if path.is_file():

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(
                f"Unsupported image format: {path.suffix}"
            )

        return [path]

    # Folder
    if path.is_dir():

        images = [
            p for p in path.iterdir()
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
        ]

        return sorted(images)

    raise FileNotFoundError(
        f"Input does not exist: {input_path}"
    )


# ============================================================
# VALID / INVALID OUTPUT FOLDERS
# (ported from the sort_plates.py reference script)
# ============================================================

def unique_path(dir_path, name):
    """Avoid clobbering an existing file by appending _1, _2, ... before
    the extension if `name` already exists in `dir_path`."""
    base, ext = os.path.splitext(name)
    out = os.path.join(dir_path, name)
    n = 1
    while os.path.exists(out):
        out = os.path.join(dir_path, f"{base}_{n}{ext}")
        n += 1
    return out


def draw_label(img, text, color):
    """
    Stamp a colored label banner across the top of the image, e.g.
    'CG04NS4270  0.54  NOT MATCH'. Shrinks the font until the text
    fits the image width (ported from sort_plates.py's draw_label,
    simplified to a single top banner since this script has no
    per-detection bounding boxes to anchor the label to).
    """
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.8, 2
    max_w = w - 20

    (tw, th), bl = cv2.getTextSize(text, font, scale, thickness)
    while tw + 15 > max_w and scale > 0.25:
        scale -= 0.05
        thickness = 1
        (tw, th), bl = cv2.getTextSize(text, font, scale, thickness)

    band_h = th + bl + 15

    # Add white padding above the image, then draw the colored label
    # band + text inside that padding so the original image is untouched.
    canvas = cv2.copyMakeBorder(
        img, band_h, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )
    cv2.rectangle(canvas, (5, 5), (min(tw + 15, w - 5), band_h - 5), color, -1)
    cv2.putText(
        canvas, text, (10, band_h - bl - 8),
        font, scale, (0, 0, 0), thickness, cv2.LINE_AA,
    )
    return canvas


def save_to_valid_invalid_folder(image_path, plate, score, is_valid, valid_dir, invalid_dir):
    """
    Draw a '<plate> <score> <VALID/NOT MATCH>' label banner on the
    image (green for valid, red for not-match), then save it into
    valid_dir or invalid_dir depending on is_valid, renaming it to
    include the recognized plate text. Returns the path saved to.
    """
    dest_dir = valid_dir if is_valid else invalid_dir

    src = Path(image_path)
    clean = "".join(c for c in (plate or "") if c.isalnum())
    suffix = clean if clean else "NONE"
    out_name = f"{src.stem}_{suffix}{src.suffix}"
    out_path = unique_path(dest_dir, out_name)

    img = cv2.imread(str(src))
    if img is None:
        # Fall back to a plain copy if OpenCV can't decode the image
        shutil.copy2(src, out_path)
        return out_path

    status = "VALID" if is_valid else "NOT MATCH"
    label_text = f"{plate or 'NONE'}  {score}  {status}"
    color = (0, 230, 80) if is_valid else (0, 0, 255)  # BGR: green / red

    annotated = draw_label(img, label_text, color)
    cv2.imwrite(str(out_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return out_path


# ============================================================
# VALID / INVALID SUMMARY
# ============================================================

def _tally(rows):
    """Shared valid/invalid tally logic, used for both per-image and
    per-detection row lists."""
    total = len(rows)
    valid_count = sum(1 for row in rows if row[4] is True)
    invalid_count = total - valid_count
    no_text_count = sum(1 for row in rows if row[1] == "")
    corrected_count = sum(1 for row in rows if row[4] is True and row[5] is True)
    source_counts = Counter(row[6] for row in rows)
    valid_pct = (valid_count / total * 100) if total else 0.0
    invalid_pct = (invalid_count / total * 100) if total else 0.0

    return {
        "total": total,
        "valid": valid_count,
        "invalid": invalid_count,
        "valid_pct": valid_pct,
        "invalid_pct": invalid_pct,
        "no_text": no_text_count,
        "corrected": corrected_count,
        "by_source": source_counts,
    }


def summarize_results(image_rows):
    """
    Tally valid vs invalid across image_rows - exactly ONE row per
    input image (the best detection for that image, or a NO TEXT
    row if nothing was detected).

    Returns a dict with counts, ready to print.
    """
    return {"per_image": _tally(image_rows)}


def print_summary(summary, valid_dir=None, invalid_dir=None):
    img = summary["per_image"]

    print("\n" + "=" * 70)
    print(f"Total images    : {img['total']}")
    print(f"Valid           : {img['valid']}  ({img['valid_pct']:.2f}%)")
    print(f"Invalid         : {img['invalid']}  ({img['invalid_pct']:.2f}%)")
    print(f"  of which no text detected : {img['no_text']}")
    print(f"Corrected (auto-fixed) : {img['corrected']}")
    print("Breakdown by validation source (best detection per image):")
    for source, count in img["by_source"].most_common():
        print(f"  {source:<16}: {count}")
    if valid_dir or invalid_dir:
        print(f"Valid images saved to   : {valid_dir}")
        print(f"Invalid images saved to : {invalid_dir}")
    print("=" * 70)


def write_folder_summary_csv(folder, rows):
    """Write a summary.csv (image, raw_text, score, plate, corrected,
    source, saved_to) inside a valid/invalid output folder."""
    path = os.path.join(folder, "summary.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "raw_text", "score", "plate", "corrected", "source", "saved_to"])
        w.writerows(rows)
    return path


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Test trained PaddleOCR license plate model"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Image file or image folder"
    )

    parser.add_argument(
        "--reference",
        default=None,
        help="Optional known-plates .txt/.csv for cross-check validation"
    )

    parser.add_argument(
        "--csv-out",
        default=None,
        help="Optional path to write one CSV with one row per image "
             "(image, raw_text, score, plate, valid, corrected, source)"
    )

    parser.add_argument(
        "--output",
        default="output",
        help="Parent folder for the valid/invalid result folders (default: output)"
    )

    parser.add_argument(
        "--valid-dir",
        default="valid",
        help="Folder name (under --output) for images with a VALID plate "
             "(default: valid)"
    )

    parser.add_argument(
        "--invalid-dir",
        default="invalid",
        help="Folder name (under --output) for images with NO MATCH "
             "(default: invalid)"
    )

    parser.add_argument(
        "--no-sort",
        action="store_true",
        help="Skip copying images into valid/invalid folders"
    )

    args = parser.parse_args()

    reference_set = load_reference_plates(args.reference) if args.reference else None
    if args.reference:
        print(f"[INFO] Loaded {len(reference_set)} reference plates from {args.reference}")

    valid_dir = os.path.join(args.output, args.valid_dir)
    invalid_dir = os.path.join(args.output, args.invalid_dir)
    if not args.no_sort:
        os.makedirs(valid_dir, exist_ok=True)
        os.makedirs(invalid_dir, exist_ok=True)

    print("Loading OCR model...")
    print(f"Model : {MODEL_DIR}")
    print(f"Dict  : {DICT_PATH}")

    ocr = create_ocr()

    images = get_images(args.input)

    print(f"\nFound {len(images)} image(s)")

    # image_rows: exactly one row per image - the best detection for
    #             that image (valid, highest score wins; else highest
    #             score overall)
    image_rows = []

    # rows collected per destination folder, for that folder's own
    # summary.csv (image, raw_text, score, plate, corrected, source, saved_to)
    rows_valid = []
    rows_invalid = []

    for index, image_path in enumerate(images, start=1):

        print(
            f"\n[{index}/{len(images)}] "
            f"{image_path.name}"
        )

        rows = process_image(ocr, image_path, reference_set=reference_set)
        best = pick_best_detection(rows)
        image_rows.append(best)

        _, raw_text, score, plate, is_valid, corrected, source = best

        # ------------------------------------------------
        # Save a copy of the image into the valid/invalid
        # output folder, named with the recognized plate.
        # ------------------------------------------------
        if not args.no_sort:
            saved_to = save_to_valid_invalid_folder(
                image_path, plate, score, is_valid, valid_dir, invalid_dir
            )
            print(f"  -> saved to: {os.path.relpath(saved_to)}")

            row = (str(image_path), raw_text, score, plate, corrected, source, saved_to)
            if is_valid:
                rows_valid.append(row)
            else:
                rows_invalid.append(row)

    if not args.no_sort:
        write_folder_summary_csv(valid_dir, rows_valid)
        write_folder_summary_csv(invalid_dir, rows_invalid)
        print(f"\n[INFO] Wrote folder summaries -> {valid_dir}/summary.csv, {invalid_dir}/summary.csv")

    if args.csv_out:
        header = ["image", "raw_text", "score", "plate", "valid", "corrected", "source"]
        with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(image_rows)
        print(f"[INFO] Wrote per-image CSV -> {args.csv_out}")

    # ------------------------------------------------
    # Valid / invalid counting (printed to console only)
    # ------------------------------------------------
    summary = summarize_results(image_rows)
    print_summary(
        summary,
        valid_dir=valid_dir if not args.no_sort else None,
        invalid_dir=invalid_dir if not args.no_sort else None,
    )


if __name__ == "__main__":
    main()