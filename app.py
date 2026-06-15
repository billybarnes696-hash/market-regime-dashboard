import os
from datetime import date, timedelta

import pandas as pd
import requests
import streamlit as st

SAM_API_URL = "https://api.sam.gov/opportunities/v2/search"

KEYWORDS = [
    "toner", "printer supplies", "office supplies", "janitorial supplies",
    "safety supplies", "PPE", "first aid kits", "furniture", "monitors",
    "docking stations", "keyboards", "mice", "batteries", "facility supplies",
    "traffic signs", "bollards", "parking blocks"
]

BAD_WORDS = [
    "construction", "staffing", "medical services", "software", "cybersecurity",
    "engineering", "architecture", "roofing", "hvac", "plumbing", "electrical",
    "security clearance", "professional services"
]

GOOD_WORDS = [
    "toner", "printer", "cartridge", "office supplies", "safety", "ppe",
    "first aid", "furniture", "monitor", "keyboard", "mouse", "battery",
    "janitorial", "delivery", "supplies"
]


def get_api_key():
    try:
        return st.secrets["SAM_API_KEY"]
    except Exception:
        return os.getenv("SAM_API_KEY")


def score_opportunity(opp):
    text = " ".join(str(v) for v in opp.values()).lower()
    score = 50

    for word in GOOD_WORDS:
        if word in text:
            score += 6

    for word in BAD_WORDS:
        if word in text:
            score -= 25

    set_aside = str(opp.get("typeOfSetAsideDescription", "")).lower()

    if "women" in set_aside or "wosb" in set_aside:
        score += 25
    elif "small business" in set_aside:
        score += 15

    title = str(opp.get("title", "")).lower()
    if any(x in title for x in ["toner", "supplies", "printer", "furniture", "safety"]):
        score += 10

    return max(min(score, 100), 0)


def categorize(opp):
    text = " ".join(str(v) for v in opp.values()).lower()

    if any(x in text for x in ["toner", "printer", "office supplies", "ppe", "safety", "first aid", "batteries"]):
        return "Drop-Ship Supply"

    if any(x in text for x in ["furniture", "equipment", "facility supplies"]):
        return "Low-Touch Supply"

    return "Review Further"


def recommendation(score):
    if score >= 85:
        return "Bid Immediately"
    if score >= 70:
        return "Review Further"
    return "Pass"


def estimate_margin(category):
    if category == "Drop-Ship Supply":
        return "5–12%"
    if category == "Low-Touch Supply":
        return "8–20%"
    return "Unknown"


def search_sam(keyword, days_back, limit=100):
    api_key = get_api_key()
    if not api_key:
        st.error("Missing SAM_API_KEY. Add it to Streamlit Secrets or environment variables.")
        st.stop()

    today = date.today()
    start = today - timedelta(days=days_back)

    params = {
        "api_key": api_key,
        "q": keyword,
        "postedFrom": start.strftime("%m/%d/%Y"),
        "postedTo": today.strftime("%m/%d/%Y"),
        "limit": limit,
        "offset": 0,
        "ptype": "o",
    }

    response = requests.get(SAM_API_URL, params=params, timeout=30)
    response.raise_for_status()
    return response.json().get("opportunitiesData", [])


st.set_page_config(page_title="Windsor Cross SAM Search", layout="wide")

st.title("Windsor Cross SAM Opportunity Search")
st.caption("Finds and ranks low-touch federal supply opportunities from SAM.gov.")

with st.sidebar:
    st.header("Search Settings")

    days_back = st.slider("Posted within last X days", 7, 90, 30)

    selected_keywords = st.multiselect(
        "Keywords",
        KEYWORDS,
        default=["toner", "office supplies", "safety supplies", "printer supplies"]
    )

    min_score = st.slider("Minimum score", 0, 100, 70)

    search_button = st.button("Search SAM.gov", type="primary")

if search_button:
    all_results = {}

    with st.spinner("Searching SAM.gov..."):
        for keyword in selected_keywords:
            try:
                results = search_sam(keyword, days_back)
                for opp in results:
                    key = opp.get("noticeId") or opp.get("solicitationNumber") or opp.get("title")
                    all_results[key] = opp
            except Exception as e:
                st.warning(f"Search failed for '{keyword}': {e}")

    rows = []

    for opp in all_results.values():
        score = score_opportunity(opp)
        category = categorize(opp)

        if score < min_score:
            continue

        rows.append({
            "Score": score,
            "Recommendation": recommendation(score),
            "Category": category,
            "Estimated Margin": estimate_margin(category),
            "Title": opp.get("title", ""),
            "Solicitation #": opp.get("solicitationNumber", ""),
            "Notice ID": opp.get("noticeId", ""),
            "Agency": opp.get("department", ""),
            "Sub Agency": opp.get("subTier", ""),
            "Set Aside": opp.get("typeOfSetAsideDescription", ""),
            "NAICS": opp.get("naicsCode", ""),
            "PSC": opp.get("classificationCode", ""),
            "Posted": opp.get("postedDate", ""),
            "Due": opp.get("responseDeadLine", ""),
            "Link": opp.get("uiLink", ""),
        })

    df = pd.DataFrame(rows)

    if df.empty:
        st.info("No matching opportunities found.")
    else:
        df = df.sort_values("Score", ascending=False).reset_index(drop=True)
        df.insert(0, "Rank", df.index + 1)

        st.subheader("Ranked Opportunities")

        st.dataframe(
            df,
            use_container_width=True,
            column_config={
                "Link": st.column_config.LinkColumn("SAM Link")
            }
        )

        st.download_button(
            "Download CSV",
            df.to_csv(index=False).encode("utf-8"),
            "windsor_cross_sam_ranked_opportunities.csv",
            "text/csv"
        )

        st.subheader("Top 10")
        st.dataframe(
            df.head(10),
            use_container_width=True,
            column_config={
                "Link": st.column_config.LinkColumn("SAM Link")
            }
        )
else:
    st.info("Select search terms and click Search SAM.gov.")
