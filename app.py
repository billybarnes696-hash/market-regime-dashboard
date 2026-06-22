import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Windsor Cross SAM Search", layout="wide")

st.title("Windsor Cross Opportunity Engine")
st.caption("Quick-hit SAM.gov search for touchless product/drop-ship opportunities")

# API key input
api_key = st.text_input(
    "Paste SAM.gov API Key",
    type="password",
    placeholder="Paste your SAM API key here"
)

# Search settings
keyword = st.text_input(
    "Search Keyword",
    value="battery OR toner OR cartridge OR gloves OR janitorial OR safety"
)

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
        "brand name",
        "authorized reseller",
        "quote",
        "delivery",
        "quantity",
        "small business set-aside"
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
        "onsite"
    ]

    for term in good_terms:
        if term in text:
            score += 5
            reasons.append(f"Positive: {term}")

    for term in bad_terms:
        if term in text:
            score -= 12
            reasons.append(f"Risk: {term}")

    return max(0, min(score, 100)), "; ".join(reasons[:6])

if st.button("Search SAM"):

    if not api_key:
        st.error("Please paste your SAM.gov API key.")
        st.stop()

    st.write("Searching SAM.gov...")

    url = "https://api.sam.gov/opportunities/v2/search"

    params = {
        "api_key": api_key,
        "q": keyword,
        "limit": limit
    }

    try:
        response = requests.get(url, params=params, timeout=30)

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
                "Type": opp.get("type", ""),
                "Set Aside": opp.get("typeOfSetAsideDescription", ""),
                "Due Date": opp.get("responseDeadLine", ""),
                "Why": reasons,
                "Link": opp.get("uiLink", "")
            })

        df = pd.DataFrame(rows)
        df = df.sort_values(by="Quick Hit Score", ascending=False)

        st.success(f"Found {len(df)} opportunities")

        st.dataframe(
            df,
            use_container_width=True,
            column_config={
                "Link": st.column_config.LinkColumn("Open in SAM")
            }
        )

    except Exception as e:
        st.error(str(e))
