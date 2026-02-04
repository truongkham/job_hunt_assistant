from crewai import Agent, Task, LLM
from utils.config import OPENAI_API_KEY, OPENAI_BASE_URL

# Groq provides an OpenAI-compatible endpoint.
llm = LLM(
    model="llama-3.1-8b-instant",
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
    temperature=0.2,
)

def get_jd_analyst_agent():
    return Agent(
        role="JD Analyst",
        goal="Understand and summarize government job postings",
        backstory="You're an expert in job market analysis with a focus on US federal job listings.",
        llm=llm,
        verbose=True,
    )

def create_jd_analysis_task(agent, job_description):
    return Task(
        description=f"""
Analyze the following USAJobs job posting and extract:

- A summary of the role
- Key responsibilities
- Key skills required
- Any specific qualifications or eligibility

Job Description:
{job_description}
""",
        expected_output=(
            "A structured markdown summary with headings:\n"
            "## Summary\n"
            "## Responsibilities\n"
            "## Required Skills\n"
            "## Qualifications / Eligibility\n"
        ),
        agent=agent,
        output_file="data/report.md",
    )
