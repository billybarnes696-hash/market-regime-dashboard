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
    "traffic signs", "bollards", "parking blocks", "medical equipment",
    "hospital beds", "patient lifts", "gurneys"
]

GOOD_WORDS = [
    "toner", "printer", "cartridge", "office supplies", "safety", "ppe",
    "first aid", "furniture", "monitor", "keyboard", "mouse", "battery",
    "janitorial", "delivery", "supplies", "medical equipment", "beds",
    "patient lifts", "gurney"
]

BAD_WORDS = [
    "construction", "staffing", "medical services", "software", "cybersecurity",
    "engineering", "architecture", "roofing", "hvac", "plumbing", "electrical",
    "security clearance", "professional services", "design-build"
]


st.set_page_config(page_title="Windsor Cross SAM Dashboard", layout="wide")

st.title("Windsor Cross SAM Opportunity Dashboard")
st.caption("Searches SAM.gov and ranks low-touch, drop-ship, WOSB-friendly opportunities.")


with st.sidebar:
    st.header("SAM API")

    api_key_input = st.text_input(
        "Paste SAM API Key",
        type="password",
        help="You can paste your SAM.gov API key here, or store it in Streamlit Secrets."
    )

    st.header("Search Filters")

    selected_keywords = st.multiselect(
        "Search Keywords",
        KEYWORDS,
        default=["toner", "office supplies", "printer supplies", "safety supplies"]
    )

    days_back = st.slider("Posted within last X days", 7, 90, 30)

    min_score = st.slider("Minimum Score", 0, 100, 70)

    wosb_only = st.checkbox("WOSB / Women-Owned Friendly Only", value=False)
    small_business_only = st.checkbox("Small Business Friendly Only", value=False)
    drop_ship_only = st.checkbox("Drop-Ship / Supply Only", value=True)

    search_button = st.button("Search SAM.gov", type="primary")


def get_api_key():
    if api_key_input:
        return api_key_input

    try:
        return st.secrets["SAM_API_KEY"]
    except Exception:
        return os.getenv("SAM_API_KEY")


def score_opportunity(opp):
    text = " ".join(str(v) for v in opp.values()).lower()
    score = 50

    for word in GOOD_WORDS:
        if word.lower() in text:
            score += 6

    for word in BAD_WORDS:
        if word.lower() in text:
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

    if any(x in text for x in [
        "toner", "printer", "office supplies", "ppe", "safety",
        "first aid", "batteries", "cartridge", "supplies"
    ]):
        return "Drop-Ship Supply"

    if any(x in text for x in [
        "furniture", "equipment", "facility supplies", "medical equipment",
        "hospital beds", "patient lifts"
    ]):
        return "Low-Touch Supply"

    return "Review Further"


def estimate_margin(category, opp):
    text = " ".join(str(v) for v in opp.values()).lower()

    if "toner" in text or "cartridge" in text:
        return "3–10%"

    if "furniture" in text:
        return "10–20%"

    if "safety" in text or "ppe" in text or "first aid" in text:
        return "8–20%"

    if "medical equipment" in text or "patient" in text or "hospital bed" in text:
        return "10–30%"

    if category == "Drop-Ship Supply":
        return "5–12%"

    if category == "Low-Touch Supply":
        return "8–20%"

    return "Unknown"


def risk_score(opp):
    text = " ".join(str(v) for v in opp.values()).lower()
    risk = 2

    if any(x in text for x in BAD_WORDS):
        risk += 5

    if any(x in text for x in ["installation", "training", "maintenance", "service"]):
        risk += 2

    if any(x in text for x in ["delivery", "supplies", "toner", "cartridge"]):
        risk -= 1

    return max(min(risk, 10), 1)


def difficulty_score(opp):
    text = " ".join(str(v) for v in opp.values()).lower()
    difficulty = 2

    if any(x in text for x in ["proposal", "technical", "management plan", "past performance"]):
        difficulty += 3

    if any(x in text for x in ["installation", "training", "maintenance"]):
        difficulty += 2

    if any(x in text for x in ["toner", "supplies", "cartridge", "delivery"]):
        difficulty -= 1

    return max(min(difficulty, 10), 1)


def recommendation(score, risk):
    if score >= 85 and risk <= 4:
        return "Bid Immediately"
    if score >= 70:
        return "Review Further"
    return "Pass"


def search_sam(keyword, days_back, limit=100):
    api_key = get_api_key()

    if not api_key:
        st.error("Missing SAM API key. Paste it in the sidebar or add it to Streamlit Secrets.")
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


if search_button:
    all_results = {}

    with st.spinner("Searching SAM.gov..."):
        for keyword in selected_keywords:
            try:
                results = search_sam(keyword, days_back)

                for opp in results:
                    key = (
                        opp.get("noticeId")
                        or opp.get("solicitationNumber")
                        or opp.get("title")
                    )
                    if key:
                        all_results[key] = opp

            except Exception as e:
                st.warning(f"Search failed for '{keyword}': {e}")

    rows = []

    for opp in all_results.values():
        score = score_opportunity(opp)
        category = categorize(opp)
        risk = risk_score(opp)
        difficulty = difficulty_score(opp)
        set_aside = str(opp.get("typeOfSetAsideDescription", ""))

        if score < min_score:
            continue

        if wosb_only and not any(x in set_aside.lower() for x in ["women", "wosb"]):
            continue

        if small_business_only and "small business" not in set_aside.lower():
            continue

        if drop_ship_only and category not in ["Drop-Ship Supply", "Low-Touch Supply"]:
            continue

        rows.append({
            "Score": score,
            "Recommendation": recommendation(score, risk),
            "Category": category,
            "Estimated Margin": estimate_margin(category, opp),
            "Risk": risk,
            "Difficulty": difficulty,
            "Title": opp.get("title", ""),
            "Solicitation #": opp.get("solicitationNumber", ""),
            "Notice ID": opp.get("noticeId", ""),
            "Agency": opp.get("department", ""),
            "Sub Agency": opp.get("subTier", ""),
            "Set Aside": set_aside,
            "NAICS": opp.get("naicsCode", ""),
            "PSC": opp.get("classificationCode", ""),
            "Posted": opp.get("postedDate", ""),
            "Due": opp.get("responseDeadLine", ""),
            "Link": opp.get("uiLink", ""),
        })

    df = pd.DataFrame(rows)

    if df.empty:
        st.info("No matching opportunities found. Try lowering the minimum score or adding more keywords.")
    else:
        df = df.sort_values("Score", ascending=False).reset_index(drop=True)
        df.insert(0, "Rank", df.index + 1)

        st.subheader("Ranked SAM Opportunities")

        st.dataframe(
            df,
            use_container_width=True,
            column_config={
                "Link": st.column_config.LinkColumn("SAM Link")
            }
        )

        st.download_button(
            "Download Full CSV",
            df.to_csv(index=False).encode("utf-8"),
            "windsor_cross_ranked_sam_opportunities.csv",
            "text/csv"
        )

        st.subheader("Top 10 Overall")
        st.dataframe(
            df.head(10),
            use_container_width=True,
            column_config={
                "Link": st.column_config.LinkColumn("SAM Link")
            }
        )

        st.subheader("Bid Immediately")
        bid_now = df[df["Recommendation"] == "Bid Immediately"]

        if bid_now.empty:
            st.info("No immediate-bid opportunities found.")
        else:
            st.dataframe(
                bid_now,
                use_container_width=True,
                column_config={
                    "Link": st.column_config.LinkColumn("SAM Link")
                }
            )

else:
    st.info("Paste your SAM API key, choose keywords, and click Search SAM.gov.")
