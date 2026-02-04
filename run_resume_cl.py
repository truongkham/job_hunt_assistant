from crewai import Crew, Process
from agents.resume_cl_agent import get_resume_cl_agent, create_resume_cl_task

job_summary = open("data/report.md", "r", encoding="utf-8").read()
resume_text = open("data/sample_resume.txt", "r", encoding="utf-8").read()

agent = get_resume_cl_agent()
task = create_resume_cl_task(agent, job_summary, resume_text)

crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=True)
result = crew.kickoff()

print("\n=== FINAL OUTPUT ===\n")
print(result)
print("\nSaved to data/resume_agent_output.txt")
