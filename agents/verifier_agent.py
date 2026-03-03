from crewai import Agent, Task
from utils.llm_factory import get_default_llm


def get_verifier_agent():
    llm = get_default_llm()
    return Agent(
        role="Application Materials Verifier",
        goal="Detect unsupported claims and improve faithfulness of application materials",
        backstory=(
            "You are a strict compliance-focused reviewer. "
            "You check that cover letters and summaries only claim what is supported by the resume."
        ),
        llm=llm,
        verbose=True,
    )


def create_verification_task(agent, resume_text: str, resume_agent_output: str, job_text: str):
    return Task(
        description=f"""
You will verify whether the generated application materials contain any claims that are NOT supported by the resume.

Inputs:

--- Resume Text (source of truth) ---
{resume_text}

--- Job Posting (context) ---
{job_text}

--- Generated Output (from Resume/Cover Letter agent) ---
{resume_agent_output}

Task:
1) Extract all specific claims made in the generated output (experience, years, tools used, achievements, education, clearances, leadership claims, etc.).
2) For each claim, decide if it is supported by the Resume Text.
3) Produce a list of unsupported claims (if any).
4) Provide a corrected version of the resume summary that removes unsupported claims (keep it 3-5 sentences).
5) Output a single "hallucination_count" number.

Output format (must be valid markdown):

## Verification Summary
- hallucination_count: <integer>
- risk_level: Low/Medium/High

## Unsupported Claims
- <claim 1> — Why unsupported
- <claim 2> — Why unsupported

## Suggested Corrected Resume Summary
<3-5 sentences, faithful to the resume>

Keep it concise and strict.
""",
        expected_output="A markdown verification report containing hallucination_count, unsupported claims, and corrected summary.",
        agent=agent,
        output_file="data/verification_report.md",
    )
