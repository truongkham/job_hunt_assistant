from crewai import Crew, Process

from agents.jd_analyst import get_jd_analyst_agent, create_jd_analysis_task
from agents.resume_cl_agent import get_resume_cl_agent, create_resume_cl_task
from agents.messaging_agent import get_messaging_agent, create_messaging_task
from utils.tracking import log_application, save_cover_letter_file


def extract_between_markers(text, start, end=None):
    try:
        start_idx = text.index(start) + len(start)
        end_idx = text.index(end, start_idx) if end else len(text)
        return text[start_idx:end_idx].strip()
    except ValueError:
        return "Not found"


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


def run_pipeline(job_data: dict, resume_text: str, user_bio: str):
    """
    job_data: USAJobs MatchedObjectDescriptor dict
    resume_text: user resume text from UI
    user_bio: short bio from UI
    Returns final crew output (string).
    """
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

    # Run the crew
    crew = Crew(
        agents=[jd_agent, resume_agent, message_agent],
        tasks=[jd_task, resume_task, message_task],
        process=Process.sequential,
        verbose=True,
    )
    result = crew.kickoff()

    # Extract key outputs from the resume agent task output
    resume_output = str(resume_task.output)
    resume_summary = extract_between_markers(resume_output, "<<RESUME_SUMMARY>>", "<<COVER_LETTER>>")
    cover_letter = extract_between_markers(resume_output, "<<COVER_LETTER>>")

    # Log and save artifacts
    log_application(job_title, agency_name, resume_summary)
    save_cover_letter_file(job_title, cover_letter)

    return str(result)