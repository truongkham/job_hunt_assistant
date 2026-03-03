from crewai import Agent, Task
from utils.llm_factory import LLMConfig, get_llm


def get_resume_cl_agent():
    # Slightly higher temperature for creativity for this agent
    llm = get_llm(LLMConfig(provider="openai_compatible", model="llama-3.1-8b-instant", temperature=0.3))
    return Agent(
        role="Resume & Cover Letter Writer",
        goal="Customize application materials to match job descriptions",
        backstory="You're an expert in professional writing and tailoring resumes for job applications, especially in government and tech roles.",
        llm=llm,
        verbose=True,
    )


def create_resume_cl_task(agent, job_summary, resume_text):
    return Task(
        description=f"""
Based on the job summary below, tailor the candidate's resume summary and generate a personalized cover letter.

--- Job Summary ---
{job_summary}

--- Resume Text ---
{resume_text}

Your output MUST include:
1. Updated professional summary for resume (3-5 sentences)
2. A personalized cover letter suitable for a government job

Return the output using the exact markers below so it can be parsed downstream.
""",
        agent=agent,
        expected_output="""
<<RESUME_SUMMARY>>
[Your tailored 3-5 sentence resume summary here]

<<COVER_LETTER>>
[Your personalized cover letter here]
""".strip(),
        output_file="data/resume_agent_output.txt",
    )
