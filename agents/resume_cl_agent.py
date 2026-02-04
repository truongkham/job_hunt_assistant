from crewai import Agent, Task, LLM
from utils.config import OPENAI_API_KEY, OPENAI_BASE_URL

# Use Groq via OpenAI-compatible endpoint (slightly higher temperature for creativity)
llm = LLM(
    model="llama-3.1-8b-instant",
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
    temperature=0.3,
)

def get_resume_cl_agent():
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
