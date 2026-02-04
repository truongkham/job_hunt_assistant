from crewai import Crew, Process
from agents.jd_analyst import get_jd_analyst_agent, create_jd_analysis_task

# simple sample description (you can replace later with a real USAJobs summary)
job_description = """
Business Analyst (Federal)
Responsibilities: gather requirements, write user stories, coordinate with stakeholders,
analyze processes, and support system improvements.
Qualifications: experience with Agile, communication skills, and data analysis.
Required skills: SQL, Excel, documentation, stakeholder management.
"""

agent = get_jd_analyst_agent()
task = create_jd_analysis_task(agent, job_description)

crew = Crew(
    agents=[agent],
    tasks=[task],
    process=Process.sequential,
    verbose=True
)

crew.kickoff()
print("Wrote report to data/report.md")
