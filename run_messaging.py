from crewai import Crew, Process
from agents.messaging_agent import get_messaging_agent, create_messaging_task

job_summary = open("data/report.md", "r", encoding="utf-8").read()
agency_name = "U.S. Customs and Border Protection"
user_bio = "I'm a data analyst skilled in SQL, Excel, reporting, and stakeholder communication. I'm passionate about public service and mission-driven work."

agent = get_messaging_agent()
task = create_messaging_task(agent, job_summary=job_summary, agency_name=agency_name, user_bio=user_bio)

crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=True)
result = crew.kickoff()

print("\n=== FINAL OUTPUT ===\n")
print(result)
print("\nSaved to data/outreach_message.txt")
