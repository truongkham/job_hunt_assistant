import csv
import re
from datetime import datetime
from pathlib import Path


DATA_DIR = Path("data")
COVER_LETTERS_DIR = DATA_DIR / "cover_letters"
LOG_PATH = DATA_DIR / "applications_log.csv"


def _safe_filename(text: str, max_len: int = 80) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9\- _]+", "", text)
    text = re.sub(r"\s+", "_", text)
    return text[:max_len].strip("_") or "cover_letter"


def save_cover_letter_file(job_title: str, cover_letter_text: str) -> str:
    """
    Saves a timestamped cover letter file into data/cover_letters/.
    Returns the saved file path as a string.
    """
    COVER_LETTERS_DIR.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{ts}_{_safe_filename(job_title)}.txt"
    path = COVER_LETTERS_DIR / filename

    path.write_text(cover_letter_text.strip() + "\n", encoding="utf-8")
    return str(path)


def log_application(job_title: str, agency_name: str, resume_summary: str) -> str:
    """
    Appends one row to data/applications_log.csv.
    Creates the file with headers if it doesn't exist.
    Returns the log path as a string.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    file_exists = LOG_PATH.exists()

    with LOG_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "job_title", "agency", "resume_summary"])
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            job_title,
            agency_name,
            resume_summary
        ])

    return str(LOG_PATH)