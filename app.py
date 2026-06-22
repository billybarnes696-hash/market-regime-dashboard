import streamlit as st
import requests
import pandas as pd
from datetime import date, timedelta

st.set_page_config(page_title="Windsor Cross SAM Search", layout="wide")

st.title("Windsor Cross Opportunity Engine")
st.caption("Quick-hit SAM.gov search for touchless product/drop-ship opportunities")

api_key = st.text_input("Paste SAM.gov API Key", type="password")

keyword = st.text_input(
    "Search Keyword",
    value="battery OR toner OR cartridge OR gloves OR janitorial OR safety"
)

col1, col2, col3 = st.columns(3)

with col1:
    posted_from = st.date_input("Posted From", date.today() - timedelta(days=30))

with col2:
    posted_to = st.date_input("Posted To", date.today())

with col3:
    limit = st.slider("Number of results", 10, 100, 25)

def quick_hit_score(text):
    text = (text or "").lower()
    score = 50
    reasons = []

    good_terms = [
        "rfq",
        "request for quote",
        "firm fixed price",
        "fob destination",
        "commercial item",
        "commercial products",
        "brand name",
        "brand name or equal",
        "authorized reseller",
        "quote",
        "delivery",
        "quantity",
        "small business set-aside",
        "total small business",
        "wosb",
        "women-owned",
        "simplified acquisition"
    ]

    bad_terms = [
        "oral presentation",
        "site visit",
        "bafo",
        "best and final",
        "technical proposal",
        "key personnel",
        "past performance",
        "security clearance",
        "facility clearance",
        "statement of work",
        "performance work statement",
        "labor categories",
        "in-person",
        "onsite",
        "on-site",
        "staffing",
        "resume",
        "subcontracting plan"
    ]

    for term in good_terms:
        if term in text:
            score += 5
            reasons.append(f"Positive: {term}")

    for term in bad_terms:
        if term in text:
            score -= 12
            reasons.append(f"Risk: {term}")

    score = max(0, min(score, 100))

    if not reasons:
        reasons.append("No major quick-hit signals found in summary")

    return score, "; ".join(reasons[:8])

if st.button("Search SAM"):

    if not api_key:
        st.error("Please paste your SAM.gov API key.")
        st.stop()

    if posted_from > posted_to:
        st.error("Posted From date cannot be after Posted To date.")
        st.stop()

    st.write("Searching SAM.gov...")

    url = "https://api.sam.gov/opportunities/v2/search"

    params = {
        "api_key": api_key,
        "q": keyword,
        "limit": limit,
        "postedFrom": posted_from.strftime("%m/%d/%Y"),
        "postedTo": posted_to.strftime("%m/%d/%Y")
    }

    try:
        response = requests.get(url, params=params, timeout=30)

        st.write(f"Status Code: {response.status_code}")

        if response.status_code != 200:
            st.error("SAM.gov API error")
            st.text(response.text)
            st.stop()

        data = response.json()
        opportunities = data.get("opportunitiesData", [])

        if not opportunities:
            st.warning("No opportunities returned.")
            st.json(data)
            st.stop()

        rows = []

        for opp in opportunities:
            title = opp.get("title", "")
            description = opp.get("description", "")
            combined_text = f"{title} {description}"

            score, reasons = quick_hit_score(combined_text)

            rows.append({
                "Quick Hit Score": score,
                "Title": title,
                "Notice ID": opp.get("noticeId", ""),
                "Agency": opp.get("department", ""),
                "Sub Agency": opp.get("subTier", ""),
                "Office": opp.get("office", ""),
                "Notice Type": opp.get("type", ""),
                "Set Aside": opp.get("typeOfSetAsideDescription", ""),
                "Posted Date": opp.get("postedDate", ""),
                "Due Date": opp.get("responseDeadLine", ""),
                "Why": reasons,
                "SAM Link": opp.get("uiLink", "")
            })

        df = pd.DataFrame(rows)
        df = df.sort_values(by="Quick Hit Score", ascending=False)

        st.success(f"Found {len(df)} opportunities")

        st.dataframe(
            df,
            use_container_width=True,
            column_config={
                "SAM Link": st.column_config.LinkColumn("Open in SAM")
            }
        )

    except Exception as e:
        st.error(str(e))
