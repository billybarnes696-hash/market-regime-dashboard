
import streamlit as st
import requests
import pandas as pd
from datetime import date, timedelta
from io import BytesIO
import zipfile
import re
import html
from urllib.parse import urlparse

try:
    import PyPDF2
except Exception:
    PyPDF2 = None

try:
    import docx
except Exception:
    docx = None

try:
    import openpyxl
except Exception:
    openpyxl = None


# ------------------------------------------------------------
# Page
# ------------------------------------------------------------
st.set_page_config(
    page_title="Windsor Cross SAM Quick-Hit Engine",
    layout="wide"
)

st.title("Windsor Cross SAM Quick-Hit Engine")
st.caption(
    "Control-panel search for low-touch, product/drop-ship RFQs. "
    "Ranks opportunities by ease of response, reseller fit, and proposal risk."
)


# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------
NOTICE_TYPE_CODES = {
    "Solicitation": "o",
    "Combined Synopsis/Solicitation": "k",
    "Pre-solicitation": "p",
    "Sources Sought": "r",
    "Award Notice": "a",
    "Special Notice": "s",
}

DEFAULT_KEYWORDS = (
    "battery OR batteries OR toner OR cartridge OR gloves OR janitorial "
    "OR cleaning OR safety OR office supplies OR paper towels OR ppe"
)

PRODUCT_CATEGORY_KEYWORDS = {
    "Batteries / Power": ["battery", "batteries", "power supply", "charger", "adapter"],
    "Toner / Printer": ["toner", "cartridge", "printer", "drum", "ink"],
    "Gloves / PPE": ["gloves", "ppe", "nitrile", "safety glasses", "mask", "respirator"],
    "Janitorial / Cleaning": ["janitorial", "cleaning", "disinfectant", "paper towels", "trash bags", "wipes"],
    "Office Supplies": ["office supplies", "paper", "folders", "pens", "staples", "binders"],
    "IT Peripherals": ["monitor", "keyboard", "mouse", "dock", "cable", "adapter", "headset"],
    "Safety / Industrial": ["safety", "first aid", "hard hat", "earplug", "hazmat", "tool"],
    "Uniforms / Apparel": ["uniform", "apparel", "shirt", "pants", "boots", "coveralls"],
}

EASY_RESPONSE_TERMS = {
    "rfq": 14,
    "request for quote": 14,
    "quote": 8,
    "quotes shall be emailed": 12,
    "email your quote": 12,
    "firm fixed price": 12,
    "ffp": 10,
    "fob destination": 8,
    "commercial product": 10,
    "commercial item": 10,
    "brand name": 8,
    "brand name or equal": 8,
    "authorized reseller": 16,
    "authorized distributor": 16,
    "lowest price": 10,
    "lowest priced technically acceptable": 10,
    "lpta": 8,
    "delivery": 5,
    "lead time": 5,
    "quantity": 5,
    "purchase order": 8,
    "simplified acquisition": 10,
    "supplies": 8,
    "product": 6,
    "unit price": 6,
    "total price": 6,
}

SET_ASIDE_TERMS = {
    "women-owned": 18,
    "wosb": 18,
    "edwosb": 18,
    "economically disadvantaged women-owned": 18,
    "total small business": 10,
    "small business set-aside": 10,
    "small business": 6,
}

HARD_RESPONSE_TERMS = {
    "oral presentation": -35,
    "in-person presentation": -35,
    "site visit": -28,
    "mandatory site visit": -40,
    "bafo": -22,
    "best and final": -22,
    "technical proposal": -30,
    "technical volume": -30,
    "management approach": -20,
    "quality control plan": -18,
    "staffing plan": -28,
    "key personnel": -30,
    "resume": -18,
    "past performance volume": -22,
    "past performance questionnaire": -22,
    "cpars": -15,
    "security clearance": -40,
    "facility clearance": -40,
    "secret clearance": -40,
    "statement of work": -14,
    "performance work statement": -18,
    "pws": -12,
    "labor categories": -30,
    "staffing": -28,
    "onsite": -22,
    "on-site": -22,
    "installation": -20,
    "maintenance services": -20,
    "service technician": -25,
    "subcontracting plan": -28,
    "bond": -30,
    "bid guarantee": -30,
    "construction": -40,
    "renovation": -40,
    "service contract act": -25,
    "wage determination": -20,
    "calibration": -35,
    "tmde": -35,
    "repair": -25,
    "design-build": -40,
    "janitorial services": -25,
    "custodial services": -25,
}

RESPONSE_REQUIREMENT_PATTERNS = {
    "CAGE / UEI required": r"\b(cage|uei|unique entity id|sam registration)\b",
    "Delivery lead time requested": r"\b(lead time|delivery schedule|delivery date|days aro|after receipt of order)\b",
    "Authorized reseller letter": r"\b(authorized reseller|authorized distributor|reseller letter|authorization letter)\b",
    "Quote by email": r"\b(email.*quote|quote.*email|send.*quote|submit.*quote.*email)\b",
    "Lowest price / LPTA": r"\b(lowest price|lpta|lowest priced technically acceptable)\b",
    "SF 1449 / Clauses": r"\b(sf\s*1449|far|dfars|representations and certifications)\b",
    "Technical narrative": r"\b(technical proposal|technical volume|technical approach|management approach)\b",
    "Past performance": r"\b(past performance|cpars|references)\b",
    "Site visit": r"\b(site visit|walk.?through)\b",
    "Oral presentation": r"\b(oral presentation|presentation)\b",
}


# ------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------
def clean_text(value):
    if value is None:
        return ""
    value = html.unescape(str(value))
    return re.sub(r"\s+", " ", value).strip()


def normalize_text(value):
    return clean_text(value).lower()


def first_value(opp, keys):
    for key in keys:
        value = opp.get(key)
        if value not in (None, "", [], {}):
            return value
    return ""


def display_notice_type(opp):
    return clean_text(first_value(opp, ["type", "noticeType", "opportunityType"]))


def display_published_date(opp):
    return clean_text(first_value(opp, ["publishedDate", "postedDate", "publishDate", "modifiedDate", "lastModifiedDate"]))


def display_due_date(opp):
    return clean_text(first_value(opp, ["responseDeadLine", "responseDeadline", "offerDueDate", "dateOffersDue"]))


def display_set_aside(opp):
    return clean_text(first_value(opp, ["typeOfSetAsideDescription", "setAsideDescription", "setAside"]))


def display_naics(opp):
    return clean_text(first_value(opp, ["naicsCode", "naics"]))


def display_psc(opp):
    return clean_text(first_value(opp, ["classificationCode", "psc", "productServiceCode"]))


def is_award_notice(notice_type):
    return "award" in notice_type.lower()


def is_sources_sought(notice_type):
    return "sources sought" in notice_type.lower()


def is_presolicitation(notice_type):
    return "pre" in notice_type.lower() and "solicitation" in notice_type.lower()


def parse_money(value):
    if value in (None, "", [], {}):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value)
    matches = re.findall(r"[\$]?\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)", text)
    if not matches:
        return None
    try:
        return float(matches[0].replace(",", ""))
    except Exception:
        return None


def get_award_amount(opp):
    # Mostly available only on award notices. Solicitations often have no reliable value.
    candidates = [
        opp.get("awardAmount"),
        opp.get("awardValue"),
        opp.get("estimatedValue"),
        opp.get("baseAndAllOptionsValue"),
    ]

    award = opp.get("award")
    if isinstance(award, dict):
        candidates.extend([
            award.get("amount"),
            award.get("awardAmount"),
            award.get("awardValue"),
        ])

    for value in candidates:
        parsed = parse_money(value)
        if parsed is not None:
            return parsed

    # Last resort: try text fields, but avoid being too aggressive.
    return None


def parse_due_date(value):
    if not value:
        return pd.NaT
    text = clean_text(value)
    # Remove common timezone abbreviations that can confuse pandas.
    text = re.sub(r"\b(EST|EDT|CST|CDT|MST|MDT|PST|PDT|GMT|UTC)\b", "", text).strip()
    try:
        return pd.to_datetime(text, errors="coerce")
    except Exception:
        return pd.NaT


def days_until_due(value):
    dt = parse_due_date(value)
    if pd.isna(dt):
        return None
    return (dt.date() - date.today()).days


def selected_category_terms(selected_categories):
    terms = []
    for cat in selected_categories:
        terms.extend(PRODUCT_CATEGORY_KEYWORDS.get(cat, []))
    return terms


def contains_any(text, terms):
    text = text.lower()
    return any(t.lower() in text for t in terms if t.strip())


def extract_emails(text):
    emails = sorted(set(re.findall(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", text or "")))
    return ", ".join(emails[:5])


def matched_requirements(text):
    text = text.lower()
    found = []
    for label, pattern in RESPONSE_REQUIREMENT_PATTERNS.items():
        if re.search(pattern, text, flags=re.IGNORECASE):
            found.append(label)
    return "; ".join(found)


def classify_response_difficulty(text, notice_type):
    t = normalize_text(text)
    hard_hits = []
    easy_hits = []

    for term in EASY_RESPONSE_TERMS:
        if term in t:
            easy_hits.append(term)

    for term in HARD_RESPONSE_TERMS:
        if term in t:
            hard_hits.append(term)

    if is_award_notice(notice_type):
        return "Ignore", "Award notice"

    hard_count = len(hard_hits)
    easy_count = len(easy_hits)

    if any(x in t for x in ["security clearance", "facility clearance", "mandatory site visit", "construction", "renovation", "design-build"]):
        return "Hard / Pass", "Hard-stop term found"

    if any(x in t for x in ["authorized reseller", "authorized distributor", "brand name"]) and easy_count >= 2 and hard_count <= 1:
        return "Easy Supplier Check", "Likely simple product quote; supplier authorization/pricing is main issue"

    if hard_count >= 5:
        return "Hard / Pass", "Multiple proposal/onsite/service-heavy requirements"

    if hard_count >= 3:
        return "Medium-Hard", "Manual review needed; several response burdens found"

    if easy_count >= 5 and hard_count <= 1:
        return "Easy Quote", "Likely straightforward RFQ"

    if easy_count >= 3 and hard_count <= 2:
        return "Medium", "Could be manageable; review attachments"

    return "Unknown / Review", "Not enough text to determine"


# ------------------------------------------------------------
# Attachment Extraction
# ------------------------------------------------------------
def extract_pdf_text(content, max_pages=20):
    if not PyPDF2:
        return ""
    try:
        reader = PyPDF2.PdfReader(BytesIO(content))
        pages = []
        for page in reader.pages[:max_pages]:
            pages.append(page.extract_text() or "")
        return "\n".join(pages)
    except Exception:
        return ""


def extract_docx_text(content):
    if not docx:
        return ""
    try:
        d = docx.Document(BytesIO(content))
        return "\n".join([p.text for p in d.paragraphs])
    except Exception:
        return ""


def extract_xlsx_text(content):
    if not openpyxl:
        return ""
    try:
        wb = openpyxl.load_workbook(BytesIO(content), read_only=True, data_only=True)
        chunks = []
        for ws in wb.worksheets[:5]:
            chunks.append(f"Sheet: {ws.title}")
            for row in ws.iter_rows(max_row=75, values_only=True):
                line = " ".join([str(c) for c in row if c is not None])
                if line.strip():
                    chunks.append(line)
        return "\n".join(chunks)
    except Exception:
        return ""


def extract_zip_text(content, max_pages=20):
    chunks = []
    try:
        with zipfile.ZipFile(BytesIO(content)) as z:
            for name in z.namelist()[:20]:
                lower = name.lower()
                try:
                    data = z.read(name)
                except Exception:
                    continue

                chunks.append(f"\n--- FILE: {name} ---\n")

                if lower.endswith(".pdf"):
                    chunks.append(extract_pdf_text(data, max_pages=max_pages))
                elif lower.endswith(".docx"):
                    chunks.append(extract_docx_text(data))
                elif lower.endswith(".xlsx"):
                    chunks.append(extract_xlsx_text(data))
                elif lower.endswith((".txt", ".csv")):
                    try:
                        chunks.append(data.decode("utf-8", errors="ignore"))
                    except Exception:
                        pass
    except Exception:
        return ""
    return "\n".join([c for c in chunks if c])


def get_resource_links(opp):
    links = opp.get("resourceLinks", []) or []
    output = []

    if isinstance(links, list):
        for link in links:
            if isinstance(link, dict):
                url = link.get("url") or link.get("href") or link.get("link")
                name = link.get("name") or link.get("filename") or url
            else:
                url = str(link)
                name = url
            if url:
                output.append({"name": name, "url": url})

    return output


@st.cache_data(show_spinner=False, ttl=900)
def download_and_extract_resource(url, api_key, max_mb, max_pages):
    try:
        params = {}
        if "api_key=" not in url:
            params["api_key"] = api_key

        r = requests.get(url, params=params, timeout=45)
        if r.status_code != 200:
            return "", f"Could not download attachment. Status {r.status_code}"

        if len(r.content) > max_mb * 1024 * 1024:
            return "", f"Skipped attachment over {max_mb} MB"

        lower = url.lower()
        content_type = r.headers.get("content-type", "").lower()

        if ".zip" in lower or "zip" in content_type:
            return extract_zip_text(r.content, max_pages=max_pages), "ZIP scanned"
        if ".pdf" in lower or "pdf" in content_type:
            return extract_pdf_text(r.content, max_pages=max_pages), "PDF scanned"
        if ".docx" in lower:
            return extract_docx_text(r.content), "DOCX scanned"
        if ".xlsx" in lower:
            return extract_xlsx_text(r.content), "XLSX scanned"
        if "text" in content_type:
            return r.text, "Text scanned"

        return "", "Attachment type not readable"
    except Exception as e:
        return "", f"Attachment error: {e}"


def get_attachment_text(opp, api_key, max_attachments, max_mb, max_pages):
    links = get_resource_links(opp)
    chunks = []
    statuses = []

    for link in links[:max_attachments]:
        text, status = download_and_extract_resource(link["url"], api_key, max_mb, max_pages)
        statuses.append(f"{link['name']}: {status}")
        if text:
            chunks.append(text)

    return "\n".join(chunks), statuses


# ------------------------------------------------------------
# Scoring
# ------------------------------------------------------------
def score_opportunity(opp, attachment_text, selected_categories, include_terms, exclude_terms):
    title = clean_text(opp.get("title", ""))
    description = clean_text(opp.get("description", ""))
    notice_type = display_notice_type(opp)
    set_aside = display_set_aside(opp)
    due = display_due_date(opp)

    combined = f"{title} {description} {notice_type} {set_aside} {attachment_text}"
    text = combined.lower()

    score = 42
    positives = []
    risks = []

    # Notice type
    if is_award_notice(notice_type):
        score -= 90
        risks.append("Award notice — already awarded/not a quote target")
    elif is_sources_sought(notice_type):
        score -= 35
        risks.append("Sources sought — market research, not quick revenue")
    elif is_presolicitation(notice_type):
        score -= 18
        risks.append("Pre-solicitation — monitor, not quote-ready")
    elif "combined" in notice_type.lower():
        score += 16
        positives.append("Combined synopsis/solicitation")
    elif "solicitation" in notice_type.lower():
        score += 12
        positives.append("Solicitation")

    # Set-aside
    for term, pts in SET_ASIDE_TERMS.items():
        if term in text:
            score += pts
            positives.append(term)

    if "no set aside" in set_aside.lower():
        score -= 12
        risks.append("No set-aside")

    # Easy and hard terms
    for term, pts in EASY_RESPONSE_TERMS.items():
        if term in text:
            score += pts
            positives.append(term)

    for term, pts in HARD_RESPONSE_TERMS.items():
        if term in text:
            score += pts
            risks.append(term)

    # Category match
    category_terms = selected_category_terms(selected_categories)
    category_hits = [t for t in category_terms if t.lower() in text]
    if category_hits:
        score += min(28, len(category_hits) * 5)
        positives.append("Category match: " + ", ".join(category_hits[:5]))
    elif selected_categories:
        score -= 15
        risks.append("No selected product category match")

    # User include/exclude terms
    include_list = [x.strip().lower() for x in re.split(r"[,;\n]", include_terms or "") if x.strip()]
    exclude_list = [x.strip().lower() for x in re.split(r"[,;\n]", exclude_terms or "") if x.strip()]

    include_hits = [x for x in include_list if x in text]
    exclude_hits = [x for x in exclude_list if x in text]

    if include_hits:
        score += min(20, len(include_hits) * 5)
        positives.append("Include hits: " + ", ".join(include_hits[:5]))

    if exclude_hits:
        score -= min(50, len(exclude_hits) * 15)
        risks.append("Exclude hits: " + ", ".join(exclude_hits[:5]))

    # Due date urgency
    dleft = days_until_due(due)
    if dleft is not None:
        if dleft < 0:
            score -= 80
            risks.append("Past due")
        elif dleft <= 2:
            score -= 25
            risks.append("Due in 0–2 days")
        elif dleft <= 5:
            score -= 10
            risks.append("Due soon")
        elif dleft >= 7:
            score += 5
            positives.append("Enough time to quote")

    # Attachments
    if attachment_text:
        positives.append("Attachments scanned")
    else:
        risks.append("No attachment text scanned")

    difficulty, difficulty_reason = classify_response_difficulty(combined, notice_type)

    # Difficulty adjustment
    if difficulty == "Easy Quote":
        score += 12
    elif difficulty == "Easy Supplier Check":
        score += 10
    elif difficulty == "Medium-Hard":
        score -= 15
    elif difficulty == "Hard / Pass":
        score -= 35
    elif difficulty == "Ignore":
        score -= 90

    score = max(0, min(100, score))

    if difficulty == "Ignore":
        action = "Ignore"
    elif difficulty == "Hard / Pass" or score < 35:
        action = "Pass"
    elif score >= 82 and difficulty in ["Easy Quote", "Easy Supplier Check", "Medium"]:
        action = "Pursue First"
    elif score >= 68:
        action = "Supplier Check"
    elif score >= 50:
        action = "Manual Review"
    else:
        action = "Pass"

    return {
        "score": score,
        "action": action,
        "difficulty": difficulty,
        "difficulty_reason": difficulty_reason,
        "positive_signals": "; ".join(dict.fromkeys(positives[:10])),
        "risks": "; ".join(dict.fromkeys(risks[:10])),
        "response_requirements": matched_requirements(combined),
        "contact_emails": extract_emails(combined),
        "days_left": days_until_due(due),
    }


# ------------------------------------------------------------
# API Search
# ------------------------------------------------------------
def build_ptype(selected_notice_types):
    codes = [NOTICE_TYPE_CODES[name] for name in selected_notice_types if name in NOTICE_TYPE_CODES]
    return ",".join(codes) if codes else "o,k"


@st.cache_data(show_spinner=False, ttl=600)
def sam_search(api_key, keyword, posted_from_str, posted_to_str, limit, ptype):
    url = "https://api.sam.gov/opportunities/v2/search"
    params = {
        "api_key": api_key,
        "q": keyword,
        "limit": limit,
        "postedFrom": posted_from_str,
        "postedTo": posted_to_str,
        "ptype": ptype,
    }
    response = requests.get(url, params=params, timeout=60)
    return response.status_code, response.text, response.json() if response.status_code == 200 else None


# ------------------------------------------------------------
# UI Control Panels
# ------------------------------------------------------------
with st.sidebar:
    st.header("1) SAM Search")

    api_key = st.text_input("Paste SAM.gov API Key", type="password")

    keyword = st.text_area(
        "Keyword Search",
        value=DEFAULT_KEYWORDS,
        height=95,
        help="Use broad product terms here. The dashboard filters/ranks after the API returns results."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        posted_from = st.date_input("Posted From", date.today() - timedelta(days=30))
    with col_b:
        posted_to = st.date_input("Posted To", date.today())

    limit = st.slider("SAM Result Limit", 10, 100, 50)

    selected_notice_types = st.multiselect(
        "Notice Types",
        options=list(NOTICE_TYPE_CODES.keys()),
        default=["Solicitation", "Combined Synopsis/Solicitation"],
        help="For quick revenue, keep Award Notices and Sources Sought off."
    )

    st.header("2) Product Panels")

    selected_categories = st.multiselect(
        "Product Categories",
        options=list(PRODUCT_CATEGORY_KEYWORDS.keys()),
        default=["Batteries / Power", "Toner / Printer", "Gloves / PPE", "Janitorial / Cleaning", "Office Supplies", "Safety / Industrial"],
    )

    set_aside_filter = st.multiselect(
        "Set-Aside Filter",
        options=["WOSB/EDWOSB", "Total Small Business", "Small Business", "No Set-Aside", "Unknown"],
        default=["WOSB/EDWOSB", "Total Small Business", "Small Business", "Unknown"],
    )

    st.header("3) Deal Size / Timing")

    min_score = st.slider("Minimum Rank Score", 0, 100, 45)

    min_value, max_value = st.slider(
        "Award / Estimated Value Filter",
        min_value=0,
        max_value=500000,
        value=(0, 250000),
        step=5000,
        help="Open solicitations often have unknown value. Use the next checkbox to decide whether to keep unknowns."
    )

    keep_unknown_value = st.checkbox("Keep unknown-value solicitations", value=True)

    max_days_left = st.slider("Max Days Until Due", 1, 90, 45)
    min_days_left = st.slider("Minimum Days Until Due", 0, 30, 3)

    st.header("4) Include / Exclude Terms")

    include_terms = st.text_area(
        "Boost if these words appear",
        value="authorized reseller\nbrand name\nfirm fixed price\nfob destination\nquote",
        height=95
    )

    exclude_terms = st.text_area(
        "Penalize / pass if these words appear",
        value="site visit\noral presentation\nbafo\nconstruction\nrenovation\nstaffing\nsecurity clearance\nfacility clearance\ncalibration\ntechnical proposal",
        height=125
    )

    st.header("5) Attachment / PDF Scan")

    scan_attachments = st.checkbox("Open API attachment links and scan PDFs/ZIPs", value=True)
    scan_top_n = st.slider("Scan attachments for top N API results", 0, 100, 25)
    max_attachments = st.slider("Max attachments per opportunity", 0, 10, 4)
    max_attachment_mb = st.slider("Max MB per attachment", 1, 25, 8)
    max_pdf_pages = st.slider("Max PDF pages to read per PDF", 1, 50, 20)

    hide_pass = st.checkbox("Hide Pass / Ignore rows", value=True)


# ------------------------------------------------------------
# Main App
# ------------------------------------------------------------
def set_aside_bucket(set_aside):
    s = (set_aside or "").lower()
    if "women" in s or "wosb" in s or "edwosb" in s:
        return "WOSB/EDWOSB"
    if "total small business" in s:
        return "Total Small Business"
    if "small business" in s:
        return "Small Business"
    if "no set aside" in s:
        return "No Set-Aside"
    return "Unknown"


if st.button("Search, Open Attachments, and Rank", type="primary"):
    if not api_key:
        st.error("Please paste your SAM.gov API key.")
        st.stop()

    if posted_from > posted_to:
        st.error("Posted From date cannot be after Posted To date.")
        st.stop()

    ptype = build_ptype(selected_notice_types)

    with st.spinner("Searching SAM.gov..."):
        status, raw_text, data = sam_search(
            api_key=api_key,
            keyword=keyword,
            posted_from_str=posted_from.strftime("%m/%d/%Y"),
            posted_to_str=posted_to.strftime("%m/%d/%Y"),
            limit=limit,
            ptype=ptype,
        )

    if status != 200:
        st.error("SAM.gov API error")
        st.text(raw_text)
        st.stop()

    opportunities = data.get("opportunitiesData", [])

    if not opportunities:
        st.warning("No opportunities returned.")
        st.json(data)
        st.stop()

    rows = []
    attachment_cache = {}
    progress = st.progress(0)

    for i, opp in enumerate(opportunities):
        attachment_text = ""
        attachment_statuses = []
        links = get_resource_links(opp)

        if scan_attachments and i < scan_top_n:
            attachment_text, attachment_statuses = get_attachment_text(
                opp=opp,
                api_key=api_key,
                max_attachments=max_attachments,
                max_mb=max_attachment_mb,
                max_pages=max_pdf_pages,
            )

        result = score_opportunity(
            opp=opp,
            attachment_text=attachment_text,
            selected_categories=selected_categories,
            include_terms=include_terms,
            exclude_terms=exclude_terms,
        )

        set_aside = display_set_aside(opp)
        value = get_award_amount(opp)
        due_text = display_due_date(opp)

        row = {
            "Rank Score": result["score"],
            "Action": result["action"],
            "Response Difficulty": result["difficulty"],
            "Difficulty Reason": result["difficulty_reason"],
            "Title": clean_text(opp.get("title", "")),
            "Notice ID": clean_text(opp.get("noticeId", "")),
            "Notice Type": display_notice_type(opp),
            "Set Aside": set_aside,
            "Set Aside Bucket": set_aside_bucket(set_aside),
            "Award/Est. Value": value,
            "NAICS": display_naics(opp),
            "PSC": display_psc(opp),
            "Published/Posted": display_published_date(opp),
            "Due": due_text,
            "Days Left": result["days_left"],
            "Attachment Count": len(links),
            "Response Requirements": result["response_requirements"],
            "Contact Emails": result["contact_emails"],
            "Positive Signals": result["positive_signals"],
            "Risks / Pass Reasons": result["risks"],
            "SAM Link": clean_text(opp.get("uiLink", "")),
            "Attachment Status": "; ".join(attachment_statuses[:5]),
            "_attachment_preview": clean_text(attachment_text[:5000]),
            "_attachment_links": links,
        }
        rows.append(row)
        progress.progress((i + 1) / len(opportunities))

    df = pd.DataFrame(rows)

    # Client-side filters
    df = df[df["Rank Score"] >= min_score]

    if hide_pass:
        df = df[~df["Action"].isin(["Pass", "Ignore"])]

    if set_aside_filter:
        df = df[df["Set Aside Bucket"].isin(set_aside_filter)]

    if "Days Left" in df.columns:
        df = df[
            df["Days Left"].isna()
            | ((df["Days Left"] >= min_days_left) & (df["Days Left"] <= max_days_left))
        ]

    if "Award/Est. Value" in df.columns:
        if keep_unknown_value:
            df = df[
                df["Award/Est. Value"].isna()
                | ((df["Award/Est. Value"] >= min_value) & (df["Award/Est. Value"] <= max_value))
            ]
        else:
            df = df[
                df["Award/Est. Value"].notna()
                & (df["Award/Est. Value"] >= min_value)
                & (df["Award/Est. Value"] <= max_value)
            ]

    if df.empty:
        st.warning("No opportunities survived your filters. Lower the minimum score, keep unknown values, or widen the date range.")
        st.stop()

    df = df.sort_values(by=["Rank Score", "Days Left"], ascending=[False, True], na_position="last")

    # Store in session so the selected review panel stays available.
    st.session_state["ranked_df"] = df

    st.success(f"Ranked {len(df)} opportunities after filters")

    summary_cols = st.columns(5)
    summary_cols[0].metric("Pursue First", int((df["Action"] == "Pursue First").sum()))
    summary_cols[1].metric("Supplier Check", int((df["Action"] == "Supplier Check").sum()))
    summary_cols[2].metric("Manual Review", int((df["Action"] == "Manual Review").sum()))
    summary_cols[3].metric("Easy Quotes", int((df["Response Difficulty"] == "Easy Quote").sum()))
    summary_cols[4].metric("Easy Supplier Checks", int((df["Response Difficulty"] == "Easy Supplier Check").sum()))

    display_cols = [
        "Rank Score", "Action", "Response Difficulty", "Title", "Notice ID",
        "Notice Type", "Set Aside", "Award/Est. Value", "NAICS", "PSC",
        "Due", "Days Left", "Attachment Count", "Response Requirements",
        "Positive Signals", "Risks / Pass Reasons", "SAM Link"
    ]

    st.dataframe(
        df[display_cols],
        use_container_width=True,
        column_config={
            "SAM Link": st.column_config.LinkColumn("Open in SAM"),
            "Rank Score": st.column_config.ProgressColumn("Rank Score", min_value=0, max_value=100),
            "Award/Est. Value": st.column_config.NumberColumn("Award/Est. Value", format="$%d"),
        }
    )

    export_df = df.drop(columns=["_attachment_preview", "_attachment_links"], errors="ignore")
    st.download_button(
        "Download Ranked CSV",
        export_df.to_csv(index=False),
        "windsor_cross_ranked_sam_results.csv",
        "text/csv"
    )


# ------------------------------------------------------------
# Detailed Review Panel
# ------------------------------------------------------------
if "ranked_df" in st.session_state:
    df = st.session_state["ranked_df"]

    st.markdown("---")
    st.header("Review One Opportunity")

    options = [
        f"{row['Rank Score']}/100 | {row['Action']} | {row['Notice ID']} | {row['Title'][:80]}"
        for _, row in df.iterrows()
    ]

    selected = st.selectbox("Choose an opportunity to inspect", options)

    if selected:
        notice_id = selected.split("|")[2].strip()
        row = df[df["Notice ID"] == notice_id].iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rank Score", int(row["Rank Score"]))
        c2.metric("Action", row["Action"])
        c3.metric("Difficulty", row["Response Difficulty"])
        c4.metric("Days Left", "Unknown" if pd.isna(row["Days Left"]) else int(row["Days Left"]))

        st.markdown(f"### {row['Title']}")
        st.write(f"**Notice ID:** {row['Notice ID']}")
        st.write(f"**Notice Type:** {row['Notice Type']}")
        st.write(f"**Set Aside:** {row['Set Aside']}")
        st.write(f"**Due:** {row['Due']}")
        st.write(f"**NAICS / PSC:** {row['NAICS']} / {row['PSC']}")
        st.markdown(f"[Open SAM Opportunity]({row['SAM Link']})")

        st.subheader("Response Readiness")
        st.write(f"**Difficulty Reason:** {row['Difficulty Reason']}")
        st.write(f"**Response Requirements Found:** {row['Response Requirements'] or 'None detected'}")
        st.write(f"**Contacts Found:** {row['Contact Emails'] or 'None detected in summary/attachments'}")

        st.subheader("Why It Ranked This Way")
        st.write(f"**Positive Signals:** {row['Positive Signals']}")
        st.write(f"**Risks / Pass Reasons:** {row['Risks / Pass Reasons']}")

        st.subheader("Attachments")
        links = row["_attachment_links"]
        if not links:
            st.info("No API resourceLinks were returned for this opportunity. Use the SAM link above to inspect manually.")
        else:
            for link in links:
                st.markdown(f"- [{link['name']}]({link['url']})")

        with st.expander("Attachment text preview used for scoring"):
            preview = row["_attachment_preview"]
            if preview:
                st.text_area("Preview", value=preview, height=300)
            else:
                st.info("No readable attachment text was scanned for this row.")
