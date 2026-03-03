import csv
import json
import os
import time
from pathlib import Path

from orchestrator import (
    run_pipeline,
    run_single_agent_pipeline,
    build_job_text,
)

from agents.verifier_agent import get_verifier_agent, create_verification_task
from crewai import Crew, Process


def count_words(text: str) -> int:
    return len([w for w in text.strip().split() if w])


def contains_markers(text: str) -> tuple[bool, bool]:
    return ("<<RESUME_SUMMARY>>" in text, "<<COVER_LETTER>>" in text)


def load_dataset(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def extract_hallucination_count(verifier_output: str) -> int:
    """
    Extract hallucination_count from verifier output.
    Returns -1 if parsing fails.
    """
    try:
        for line in verifier_output.splitlines():
            if "hallucination_count" in line:
                # example line: - hallucination_count: 4
                parts = line.split(":")
                return int(parts[1].strip())
    except Exception:
        return -1
    return -1


if __name__ == "__main__":
    dataset_path = "datasets/usajobs_business_analyst_washington_dc_25.json"
    data = load_dataset(dataset_path)
    jobs = data["jobs"][:3]  # keep small for now

    resume_text = Path("data/sample_resume.txt").read_text(encoding="utf-8")
    user_bio = "I'm a data professional passionate about public service."

    models = [
        "llama-3.1-8b-instant",
    ]

    out_dir = Path("experiments/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_csv = out_dir / f"basic_eval_multimodel_{ts}.csv"

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "architecture",
                "model",
                "job_index",
                "title",
                "agency",
                "latency_sec",
                "outreach_word_count",
                "outreach_under_150",
                "resume_marker_present",
                "cover_letter_marker_present",
                "hallucination_count",
            ],
        )
        writer.writeheader()

        for model in models:
            os.environ["LLM_MODEL"] = model
            print("\n=== Running model:", model, "===\n")

            for architecture in ["single", "multi"]:
                print(f"\n--- Architecture: {architecture} ---\n")

                for i, rec in enumerate(jobs):
                    job_item = rec["raw"]
                    md = job_item.get("MatchedObjectDescriptor", {}) or {}

                    title = rec.get("title", "Unknown Title")
                    agency = rec.get("agency", "Unknown Agency")

                    # -------------------
                    # Run generation
                    # -------------------
                    start = time.time()

                    if architecture == "single":
                        _ = run_single_agent_pipeline(md, resume_text, user_bio)
                    else:
                        _ = run_pipeline(md, resume_text, user_bio)

                    latency = time.time() - start

                    # -------------------
                    # Load generated files
                    # -------------------
                    outreach_path = Path("data/outreach_message.txt")
                    resume_out_path = Path("data/resume_agent_output.txt")

                    outreach = (
                        outreach_path.read_text(encoding="utf-8")
                        if outreach_path.exists()
                        else ""
                    )

                    resume_out = (
                        resume_out_path.read_text(encoding="utf-8")
                        if resume_out_path.exists()
                        else ""
                    )

                    # -------------------
                    # Metrics
                    # -------------------
                    wc = count_words(outreach)
                    under_150 = wc <= 150
                    resume_marker, cover_marker = contains_markers(resume_out)

                    # -------------------
                    # Run verifier for BOTH architectures
                    # -------------------
                    job_text, _, _ = build_job_text(md)

                    verifier_agent = get_verifier_agent()
                    verify_task = create_verification_task(
                        verifier_agent,
                        resume_text,
                        resume_out,
                        job_text,
                    )

                    verifier_crew = Crew(
                        agents=[verifier_agent],
                        tasks=[verify_task],
                        process=Process.sequential,
                        verbose=False,
                    )

                    verifier_result = verifier_crew.kickoff()
                    hallucination_count = extract_hallucination_count(
                        str(verifier_result)
                    )

                    # -------------------
                    # Write CSV row
                    # -------------------
                    writer.writerow(
                        {
                            "architecture": architecture,
                            "model": model,
                            "job_index": i,
                            "title": title,
                            "agency": agency,
                            "latency_sec": round(latency, 2),
                            "outreach_word_count": wc,
                            "outreach_under_150": under_150,
                            "resume_marker_present": resume_marker,
                            "cover_letter_marker_present": cover_marker,
                            "hallucination_count": hallucination_count,
                        }
                    )

                    print(
                        f"[{architecture}] [{i+1}/{len(jobs)}] "
                        f"{title} | wc={wc} | under150={under_150} "
                        f"| hallucinations={hallucination_count} "
                        f"| latency={latency:.2f}s"
                    )

    print("\nWrote results to:", out_csv)
