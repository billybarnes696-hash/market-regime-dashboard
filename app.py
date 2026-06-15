import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Windsor Cross SAM Search")

st.title("Windsor Cross Opportunity Engine")

# API KEY BOX
api_key = st.text_input(
    "Paste SAM API Key",
    type="password",
    placeholder="Paste your SAM API key here"
)

keyword = st.text_input(
    "Search Keyword",
    value="toner"
)

if st.button("Search SAM"):

    if not api_key:
        st.error("Please paste your SAM API key.")
        st.stop()

    st.write("Searching...")

    url = "https://api.sam.gov/opportunities/v2/search"

    params = {
        "api_key": api_key,
        "q": keyword,
        "limit": 25
    }

    try:
        response = requests.get(url, params=params, timeout=30)

        st.write(f"Status Code: {response.status_code}")

        if response.status_code != 200:
            st.error(response.text)
            st.stop()

        data = response.json()

        opportunities = data.get("opportunitiesData", [])

        if not opportunities:
            st.warning("No opportunities returned.")
            st.json(data)
            st.stop()

        rows = []

        for opp in opportunities:

            rows.append({
                "Title": opp.get("title", ""),
                "Notice ID": opp.get("noticeId", ""),
                "Agency": opp.get("department", ""),
                "Due Date": opp.get("responseDeadLine", ""),
                "Link": opp.get("uiLink", "")
            })

        df = pd.DataFrame(rows)

        st.success(f"Found {len(df)} opportunities")

        st.dataframe(df)

    except Exception as e:
        st.error(str(e))
