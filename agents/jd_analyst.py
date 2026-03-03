from crewai import Agent, Task
from utils.llm_factory import get_default_llm



def get_jd_analyst_agent():
    llm = get_default_llm()
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
