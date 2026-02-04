import requests
from .config import USAJOBS_API_KEY, USAJOBS_USER_AGENT

def fetch_usajobs(keyword, location="remote", results_per_page=5):
    """
    Fetch job postings from the USAJobs API.
    Returns a list of SearchResultItems (raw USAJobs job items).
    """
    headers = {
        "Host": "data.usajobs.gov",
        "User-Agent": USAJOBS_USER_AGENT,
        "Authorization-Key": USAJOBS_API_KEY,
        "Accept": "application/json",
    }

    params = {
        "Keyword": keyword,
        "ResultsPerPage": results_per_page,
    }

    # Treat "remote" as: do not apply a LocationName filter
    if location and location.lower() != "remote":
        params["LocationName"] = location

    url = "https://data.usajobs.gov/api/search"
    response = requests.get(url, headers=headers, params=params, timeout=30)

    if response.status_code == 200:
        return response.json().get("SearchResult", {}).get("SearchResultItems", [])
    return []

if __name__ == "__main__":
    jobs = fetch_usajobs("business analyst", location="Washington, DC", results_per_page=5)
    for job in jobs:
        md = job["MatchedObjectDescriptor"]
        print(f"{md['PositionTitle']} at {md['OrganizationName']}")
