import json
from datetime import datetime
from pathlib import Path

from utils.usajobs_api import fetch_usajobs


def job_to_record(job_item: dict) -> dict:
    """Convert USAJobs SearchResultItem into a compact record for experiments."""
    md = job_item.get("MatchedObjectDescriptor", {}) or {}
    details = (md.get("UserArea", {}) or {}).get("Details", {}) or {}

    title = md.get("PositionTitle", "Unknown Title")
    agency = md.get("OrganizationName", "Unknown Agency")

    summary = (details.get("JobSummary") or "").strip()
    qual_summary = (md.get("QualificationSummary") or "").strip()
    responsibilities = (details.get("Responsibilities") or "").strip()
    requirements = (details.get("Requirements") or "").strip()

    job_text = f"""Position Title: {title}
Agency: {agency}

--- Job Summary ---
{summary}

--- Qualifications Summary ---
{qual_summary}

--- Responsibilities ---
{responsibilities}

--- Requirements ---
{requirements}
""".strip()

    return {
        "title": title,
        "agency": agency,
        "position_uri": md.get("PositionURI"),
        "position_id": md.get("PositionID"),
        "application_close_date": md.get("ApplicationCloseDate"),
        "raw": job_item,          # keep original for debugging
        "job_text": job_text,     # canonical text input for agents
    }


def build_dataset(keyword: str, location: str, results_per_page: int = 25) -> dict:
    items = fetch_usajobs(keyword, location=location, results_per_page=results_per_page)
    records = [job_to_record(it) for it in items]

    return {
        "meta": {
            "source": "USAJobs API (search endpoint)",
            "keyword": keyword,
            "location": location,
            "results_per_page": results_per_page,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "count": len(records),
        },
        "jobs": records,
    }


if __name__ == "__main__":
    out_dir = Path("datasets")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Start with one query; later you can build multiple datasets for different roles.
    data = build_dataset(keyword="business analyst", location="Washington, DC", results_per_page=25)

    out_path = out_dir / "usajobs_business_analyst_washington_dc_25.json"
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    print("Wrote:", out_path)
    print("Jobs:", data["meta"]["count"])
