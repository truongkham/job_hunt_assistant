from crewai import Crew, Process
from pathlib import Path

from agents.jd_analyst import get_jd_analyst_agent, create_jd_analysis_task
from agents.resume_cl_agent import get_resume_cl_agent, create_resume_cl_task
from agents.messaging_agent import get_messaging_agent, create_messaging_task
from agents.verifier_agent import get_verifier_agent, create_verification_task
from utils.tracking import log_application, save_cover_letter_file
from utils.llm_factory import LLMConfig, get_llm


# -----------------------------
# Helper Utilities
# -----------------------------

def extract_between_markers(text, start, end=None):
    try:
        start_idx = text.index(start) + len(start)
        end_idx = text.index(end, start_idx) if end else len(text)
        return text[start_idx:end_idx].strip()
    except ValueError:
        return "Not found"


def enforce_word_limit_on_file(filepath: str, max_words: int = 150):
    """
    Truncate file content to max_words if it exceeds limit.
    Used for outreach enforcement.
    """
    path = Path(filepath)
    if not path.exists():
        return

    text = path.read_text(encoding="utf-8")
    words = text.split()

    if len(words) > max_words:
        truncated = " ".join(words[:max_words])
        path.write_text(truncated, encoding="utf-8")


def build_job_text(job_data: dict) -> tuple[str, str, str]:
    """Return (job_text, agency_name, job_title) from a USAJobs MatchedObjectDescriptor dict."""
    details = (job_data.get("UserArea", {}) or {}).get("Details", {}) or {}

    summary = (details.get("JobSummary") or "").strip()
    qual_summary = (job_data.get("QualificationSummary") or "").strip()
    responsibilities = (details.get("Responsibilities") or "").strip()
    requirements = (details.get("Requirements") or "").strip()

    agency = job_data.get("OrganizationName", "Unknown Agency")
    title = job_data.get("PositionTitle", "Unknown Position")

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

    return job_text, agency, title


# -----------------------------
# MULTI-AGENT PIPELINE
# -----------------------------

def run_pipeline(job_data: dict, resume_text: str, user_bio: str):
    if not resume_text or not resume_text.strip():
        return "Resume text is empty."

    if not user_bio or not user_bio.strip():
        user_bio = "I'm a data professional passionate about public service."

    job_text, agency_name, job_title = build_job_text(job_data)

    # Initialize agents
    jd_agent = get_jd_analyst_agent()
    resume_agent = get_resume_cl_agent()
    message_agent = get_messaging_agent()

    # Create tasks
    jd_task = create_jd_analysis_task(jd_agent, job_text)
    resume_task = create_resume_cl_task(resume_agent, job_text, resume_text)
    message_task = create_messaging_task(message_agent, job_text, agency_name, user_bio)

    # Run crew
    crew = Crew(
        agents=[jd_agent, resume_agent, message_agent],
        tasks=[jd_task, resume_task, message_task],
        process=Process.sequential,
        verbose=True,
    )
    result = crew.kickoff()

    # -----------------------------
    # Enforce Outreach Word Limit
    # -----------------------------
    enforce_word_limit_on_file("data/outreach_message.txt", max_words=150)

    # -----------------------------
    # Run Verifier
    # -----------------------------
    resume_output = str(resume_task.output)

    verifier_agent = get_verifier_agent()
    verify_task = create_verification_task(
        verifier_agent,
        resume_text,
        resume_output,
        job_text,
    )

    verifier_crew = Crew(
        agents=[verifier_agent],
        tasks=[verify_task],
        process=Process.sequential,
        verbose=True,
    )
    _ = verifier_crew.kickoff()

    # Extract resume components
    resume_summary = extract_between_markers(
        resume_output,
        "<<RESUME_SUMMARY>>",
        "<<COVER_LETTER>>"
    )
    cover_letter = extract_between_markers(
        resume_output,
        "<<COVER_LETTER>>"
    )

    # Log & Save
    log_application(job_title, agency_name, resume_summary)
    save_cover_letter_file(job_title, cover_letter)

    return str(result)


# -----------------------------
# SINGLE-AGENT BASELINE
# -----------------------------

def run_single_agent_pipeline(job_data: dict, resume_text: str, user_bio: str):
    if not resume_text or not resume_text.strip():
        return "Resume text is empty."

    job_text, agency_name, job_title = build_job_text(job_data)

    llm = get_llm(
        LLMConfig(
            provider="openai_compatible",
            model="llama-3.1-8b-instant",
            temperature=0.3,
        )
    )

    prompt = f"""
You are an AI job application assistant.

Analyze the job description and generate:

1. A short job analysis (3-5 bullet points)
2. Updated resume summary (3-5 sentences)
3. Personalized cover letter
4. Outreach message (UNDER 150 words)

STRICT REQUIREMENTS:
- Outreach must be <= 150 words
- Follow exact markers below

--- Job Description ---
{job_text}

--- Resume ---
{resume_text}

--- User Bio ---
{user_bio}

Return output using EXACT format:

<<JOB_ANALYSIS>>
...

<<RESUME_SUMMARY>>
...

<<COVER_LETTER>>
...

<<OUTREACH_MESSAGE>>
...
"""

    response = llm.call(prompt)
    response_text = str(response)

    # -----------------------------
    # Enforce Outreach Limit (Single-Agent)
    # -----------------------------
    if "<<OUTREACH_MESSAGE>>" in response_text:
        parts = response_text.split("<<OUTREACH_MESSAGE>>")
        prefix = parts[0]
        outreach_text = parts[1].strip()

        words = outreach_text.split()
        if len(words) > 150:
            outreach_text = " ".join(words[:150])

        response_text = prefix + "<<OUTREACH_MESSAGE>>\n" + outreach_text

    return response_text
