import streamlit as st

from orchestrator import run_pipeline, run_single_agent_pipeline
from utils.usajobs_api import fetch_usajobs

st.set_page_config(page_title="AI Job Hunt Assistant", layout="centered")

st.title("AI Job Hunt Assistant")
st.markdown(
    "Search USAJobs, select postings, then run AI agents to analyze the job, "
    "tailor resume + cover letter, and generate an outreach message."
)

# Inputs
keyword = st.text_input("Job Keyword", "business analyst")
location = st.text_input("Location", "Washington, DC")
resume_text = st.text_area("Paste Your Resume", height=220)
user_bio = st.text_area(
    "Short Bio (for outreach tone)",
    "I'm a data professional passionate about public service.",
    height=100,
)

mode = st.radio(
    "Select Architecture Mode:",
    ["Multi-Agent (CrewAI)", "Single-Agent (Baseline)"]
)


st.markdown("---")

# Step 1: Search jobs (store in session_state)
if st.button("Search Jobs"):
    with st.spinner("Fetching jobs from USAJobs..."):
        jobs = fetch_usajobs(keyword, location=location, results_per_page=5)

    if not jobs:
        st.error("No job postings found for this search.")
    else:
        st.session_state["jobs"] = jobs
        # reset selections
        st.session_state["selected"] = {}
        st.success("Jobs fetched! Select the ones you'd like to run the workflow for.")

# Step 2: Show list with checkboxes
if "jobs" in st.session_state:
    st.markdown("### Select Jobs to Apply For:")
    selected_indexes = []

    for i, job in enumerate(st.session_state["jobs"]):
        md = job.get("MatchedObjectDescriptor", {}) or {}
        title = md.get("PositionTitle", "Unknown Title")
        org = md.get("OrganizationName", "Unknown Agency")

        checked = st.checkbox(f"{title} — {org}", key=f"job_{i}")
        if checked:
            selected_indexes.append(i)

    st.markdown("---")

    # Step 3: Run pipeline for selected jobs
    if st.button("Apply to Selected Jobs"):
        if not selected_indexes:
            st.warning("Please select at least one job.")
        elif not resume_text.strip():
            st.warning("Please paste your resume before applying.")
        else:
            for i in selected_indexes:
                job_data = st.session_state["jobs"][i].get("MatchedObjectDescriptor", {}) or {}
                title = job_data.get("PositionTitle", "Unknown Title")
                org = job_data.get("OrganizationName", "Unknown Agency")

                st.markdown(f"## Results for: {title} — {org}")
                with st.spinner(f"Running agents for: {title}"):
                    if mode == "Single-Agent (Baseline)":
                        result = run_single_agent_pipeline(job_data, resume_text, user_bio)
                    else:
                        result = run_pipeline(job_data, resume_text, user_bio)


                st.markdown(result)
                st.markdown("---")