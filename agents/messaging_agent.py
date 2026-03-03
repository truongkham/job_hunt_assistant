from crewai import Agent, Task
from utils.llm_factory import LLMConfig, get_llm

# A bit higher temperature for friendly tone

def get_messaging_agent():
    llm = get_llm(LLMConfig(provider="openai_compatible", model="llama-3.1-8b-instant", temperature=0.5))
    return Agent(
        role="Outreach Message Writer",
        goal="Draft personalized messages for job outreach",
        backstory="You're a professional career coach skilled in writing effective cold emails and outreach messages for job seekers in tech and government.",
        llm=llm,
        verbose=True,
    )

def create_messaging_task(
    agent,
    job_summary,
    agency_name,
    user_bio="I'm a data professional passionate about public service."
):
    return Task(
        description=f"""
Write a concise and compelling outreach message that the candidate could send to someone at {agency_name}, expressing interest in the job described below.

--- Job Summary ---
{job_summary}

--- Candidate Bio ---
{user_bio}

Requirements:
- Friendly and professional tone
- Under 150 words
- Tailored for LinkedIn or email
- End with a clear call-to-action (e.g., asking for a quick chat)
""",
        expected_output=(
            "A short outreach message under 150 words, tailored for LinkedIn or email, "
            "that is professional and expresses interest in the job at the given agency."
        ),
        agent=agent,
        output_file="data/outreach_message.txt",
    )
