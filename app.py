
import streamlit as st
import requests
import pandas as pd
from datetime import date, timedelta
from io import BytesIO
import zipfile
import re
import html
from urllib.parse import quote_plus

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


# ============================================================
# Windsor Cross SAM Opportunity Funnel - v9
#
# This version is less frustrating by design:
# - Runs multiple smaller SAM searches instead of one giant OR query.
# - Shows funnel counts: raw results, category matches, quote-ready candidates.
# - Has Discovery / Balanced / Strict modes.
# - Keeps "maybe" opportunities visible in Discovery mode instead of hiding everything.
# - Still filters out obvious non-targets: awards, sources sought, construction, staffing, etc.
# ============================================================

st.set_page_config(page_title="Windsor Cross SAM Opportunity Funnel v9", layout="wide")

st.title("Windsor Cross SAM Opportunity Funnel v9")
st.caption("Find more quote-ready product opportunities first, then rank them for low-touch reseller/drop-ship fit.")


# -----------------------------
# Search Packs
# -----------------------------
SEARCH_PACKS = {
    "Toner / Printer / Office": [
        "toner", "printer cartridge", "ink cartridge", "drum cartridge",
        "office supplies", "copy paper", "paper", "binders", "envelopes"
    ],
    "Janitorial / Cleaning Supplies": [
        "janitorial supplies", "cleaning supplies", "disinfectant",
        "paper towels", "trash bags", "wipes", "soap", "sanitizer"
    ],
    "Gloves / PPE / Safety Supplies": [
        "nitrile gloves", "disposable gloves", "ppe", "safety glasses",
        "first aid kit", "ear plugs", "safety vest", "hard hats"
    ],
    "Batteries / Power": [
        "battery", "batteries", "lithium battery", "power supply",
        "charger", "ups battery", "adapter"
    ],
    "IT Peripherals": [
        "monitor", "keyboard", "mouse", "headset", "webcam",
        "docking station", "cable", "adapter", "scanner"
    ],
    "Uniforms / Apparel": [
        "uniforms", "boots", "shirts", "pants", "coveralls", "jackets"
    ],
    "Industrial Consumables": [
        "filters", "labels", "tape", "tools", "hardware", "fasteners",
        "spill kit", "cones", "signs"
    ],
}

PRODUCT_TERMS = sorted(set([t for terms in SEARCH_PACKS.values() for t in terms]))

OBVIOUS_PASS_TITLE_TERMS = [
    "construction", "renovation", "repair", "maintenance", "calibration", "tmde",
    "services", "staffing", "support services", "janitorial services", "custodial services",
    "installation", "install", "training", "software development", "content package",
    "fuel", "diesel", "gasoline", "inspection", "testing", "rental", "lease",
    "architect", "engineering", "design-build", "landscaping", "snow removal"
]

ALLOW_TITLE_TERMS = [
    "janitorial supplies", "cleaning supplies", "office supplies", "safety supplies",
    "first aid supplies", "toner", "cartridge", "gloves", "battery", "batteries",
    "paper towels", "trash bags", "monitor", "keyboard", "mouse", "uniforms"
]

EASY_TERMS = [
    "rfq", "request for quote", "quote", "quotation", "firm fixed price", "ffp",
    "fob destination", "commercial product", "commercial item", "cots", "brand name",
    "brand name or equal", "authorized reseller", "authorized distributor", "unit price",
    "total price", "delivery", "lead time", "quantity", "purchase order",
    "simplified acquisition", "lowest price", "lpta"
]

HARD_TERMS = [
    "oral presentation", "site visit", "mandatory site visit", "bafo", "best and final",
    "technical proposal", "technical volume", "management approach", "quality control plan",
    "staffing plan", "key personnel", "resume", "past performance", "security clearance",
    "facility clearance", "statement of work", "performance work statement", "pws",
    "labor categories", "onsite", "on-site", "installation", "service technician",
    "subcontracting plan", "bond", "bid guarantee", "service contract act",
    "wage determination", "construction", "renovation", "calibration", "tmde"
]

SET_ASIDE_BUCKETS = ["WOSB/EDWOSB", "Total Small Business", "Small Business", "No Set-Aside", "Unknown"]


# -----------------------------
# Helpers
# -----------------------------
def clean_text(value):
    if value is None:
        return ""
    value = html.unescape(str(value))
    return re.sub(r"\s+", " ", value).strip()


def first_value(opp, keys):
    for key in keys:
        value = opp.get(key)
        if value not in (None, "", [], {}):
            return value
    return ""


def get_notice_type(opp):
    return clean_text(first_value(opp, ["type", "noticeType", "opportunityType"]))


def get_notice_id(opp):
    return clean_text(first_value(opp, ["solicitationNumber", "noticeId", "opportunityId", "id"]))


def get_title(opp):
    return clean_text(opp.get("title", ""))


def get_description(opp):
    return clean_text(opp.get("description", ""))


def get_set_aside(opp):
    return clean_text(first_value(opp, ["typeOfSetAsideDescription", "setAsideDescription", "setAside"]))


def get_due_date(opp):
    return clean_text(first_value(opp, ["responseDeadLine", "responseDeadline", "offerDueDate", "dateOffersDue"]))


def get_posted_date(opp):
    return clean_text(first_value(opp, ["publishedDate", "postedDate", "publishDate", "modifiedDate", "lastModifiedDate"]))


def get_naics(opp):
    return clean_text(first_value(opp, ["naicsCode", "naics"]))


def get_psc(opp):
    return clean_text(first_value(opp, ["classificationCode", "psc", "productServiceCode"]))


def get_ui_link(opp):
    return clean_text(opp.get("uiLink", ""))


def parse_due_date(value):
    if not value:
        return pd.NaT
    text = clean_text(value)
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


def is_quote_ready_notice_type(notice_type):
    n = notice_type.lower()
    return ("solicitation" in n or "combined" in n) and "award" not in n and "sources sought" not in n and "pre" not in n


def is_excluded_notice_type(notice_type):
    n = notice_type.lower()
    return "award" in n or "sources sought" in n or "pre-solicitation" in n or "presolicitation" in n


def title_obvious_pass(title):
    t = title.lower()
    if any(x in t for x in ALLOW_TITLE_TERMS):
        return False
    return any(x in t for x in OBVIOUS_PASS_TITLE_TERMS)


def product_hits(text, selected_packs):
    terms = []
    for pack in selected_packs:
        terms.extend(SEARCH_PACKS.get(pack, []))
    terms = sorted(set(terms))
    t = text.lower()
    hits = []
    for term in terms:
        if re.search(rf"(?<![a-z0-9]){re.escape(term.lower())}(?![a-z0-9])", t):
            hits.append(term)
    return hits


def broad_product_hits(text):
    t = text.lower()
    hits = []
    for term in PRODUCT_TERMS:
        if re.search(rf"(?<![a-z0-9]){re.escape(term.lower())}(?![a-z0-9])", t):
            hits.append(term)
    return sorted(set(hits))


def is_service_psc(psc):
    p = clean_text(psc)
    return bool(p) and not p[0].isdigit()


def gsa_advantage_link(query):
    query = clean_text(query)
    if not query:
        return ""
    return f"https://www.gsaadvantage.gov/advantage/ws/search/advantage_search?keyword={quote_plus(query)}"


def extract_emails(text):
    return ", ".join(sorted(set(re.findall(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", text or "")))[:5])


def find_nsns(text):
    patterns = [r"\b\d{4}[-\s]?\d{2}[-\s]?\d{3}[-\s]?\d{4}\b", r"\b\d{13}\b"]
    out = []
    for p in patterns:
        out.extend(re.findall(p, text))
    return ", ".join(sorted(set(out))[:5])


def find_part_numbers(text):
    upper = text.upper()
    patterns = [
        r"\bpart(?:\s+number|\s+no\.?|#)?\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-_/\.]{2,30})\b",
        r"\bp/n\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-_/\.]{2,30})\b",
        r"\bmpn\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-_/\.]{2,30})\b",
        r"\bmodel\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-_/\.]{2,30})\b",
    ]
    out = []
    for p in patterns:
        out.extend(re.findall(p, upper, flags=re.I))
    bad = {"THE", "AND", "FOR", "WITH", "QUOTE", "SOLICITATION", "ITEM"}
    cleaned = []
    for x in out:
        x = x.strip(" .,:;()[]{}")
        if x and x not in bad:
            cleaned.append(x)
    return ", ".join(sorted(set(cleaned))[:8])


def find_quantities(text):
    patterns = [
        r"\bqty\.?\s*[:\-]?\s*(\d{1,6})\b",
        r"\bquantity\s*[:\-]?\s*(\d{1,6})\b",
        r"\b(\d{1,6})\s*(each|ea|boxes|box|units|unit|cases|case|pairs|pair)\b",
    ]
    out = []
    for p in patterns:
        for m in re.findall(p, text, flags=re.I):
            if isinstance(m, tuple):
                out.append(" ".join([str(x) for x in m if x]))
            else:
                out.append(str(m))
    return ", ".join(sorted(set(out))[:8])


def primary_query(title, hits, parts, nsns):
    if parts:
        return parts.split(",")[0].strip()
    if nsns:
        return nsns.split(",")[0].strip()
    if hits:
        return hits[0]
    return title


# -----------------------------
# Attachment extraction
# -----------------------------
def extract_pdf_text(content, max_pages=15):
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
            for row in ws.iter_rows(max_row=80, values_only=True):
                line = " ".join([str(c) for c in row if c is not None])
                if line.strip():
                    chunks.append(line)
        return "\n".join(chunks)
    except Exception:
        return ""


def extract_zip_text(content, max_pages=15):
    chunks = []
    try:
        with zipfile.ZipFile(BytesIO(content)) as z:
            for name in z.namelist()[:20]:
                try:
                    data = z.read(name)
                except Exception:
                    continue
                lower = name.lower()
                chunks.append(f"\n--- FILE: {name} ---\n")
                if lower.endswith(".pdf") or data[:4] == b"%PDF":
                    chunks.append(extract_pdf_text(data, max_pages))
                elif lower.endswith(".docx"):
                    chunks.append(extract_docx_text(data))
                elif lower.endswith(".xlsx"):
                    chunks.append(extract_xlsx_text(data))
                elif lower.endswith((".txt", ".csv")):
                    chunks.append(data.decode("utf-8", errors="ignore"))
    except Exception:
        return ""
    return "\n".join(chunks)


def get_resource_links(opp):
    links = opp.get("resourceLinks", []) or []
    out = []
    if isinstance(links, list):
        for link in links:
            if isinstance(link, dict):
                url = link.get("url") or link.get("href") or link.get("link")
                name = link.get("name") or link.get("filename") or url
            else:
                url = str(link)
                name = url
            if url:
                out.append({"name": name, "url": url})
    return out


@st.cache_data(show_spinner=False, ttl=900)
def download_and_extract(url, api_key, max_mb, max_pages):
    for params in [{}, {"api_key": api_key}]:
        try:
            r = requests.get(url, params=params, timeout=40, headers={"User-Agent": "WindsorCrossFunnel/1.0"})
            if r.status_code != 200:
                continue
            if len(r.content) > max_mb * 1024 * 1024:
                return "", f"Skipped > {max_mb}MB"
            content = r.content
            ct = r.headers.get("content-type", "").lower()
            lower = url.lower()
            if content[:4] == b"%PDF" or ".pdf" in lower or "pdf" in ct:
                txt = extract_pdf_text(content, max_pages)
                return txt, "PDF scanned" if txt else "PDF no text"
            if content[:2] == b"PK" or ".zip" in lower or "zip" in ct:
                txt = extract_zip_text(content, max_pages)
                return txt, "ZIP/Office scanned" if txt else "ZIP no text"
            if ".docx" in lower:
                return extract_docx_text(content), "DOCX scanned"
            if ".xlsx" in lower:
                return extract_xlsx_text(content), "XLSX scanned"
            if "text" in ct or "json" in ct or "xml" in ct:
                return r.text, "Text scanned"
        except Exception as e:
            return "", f"Attachment error: {e}"
    return "", "No readable attachment"


def get_attachment_text(opp, api_key, max_attachments, max_mb, max_pages):
    chunks, statuses = [], []
    for link in get_resource_links(opp)[:max_attachments]:
        txt, status = download_and_extract(link["url"], api_key, max_mb, max_pages)
        statuses.append(f"{link['name']}: {status}")
        if txt:
            chunks.append(txt)
    return "\n".join(chunks), "; ".join(statuses[:5])


# -----------------------------
# Pricing upload
# -----------------------------
def prepare_price_df(uploaded):
    if uploaded is None:
        return pd.DataFrame()
    try:
        if uploaded.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
    except Exception:
        return pd.DataFrame()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    part_col = next((c for c in ["part_number", "part", "sku", "mpn", "manufacturer_part_number", "nsn"] if c in df.columns), None)
    cost_col = next((c for c in ["unit_cost", "cost", "price", "your_cost", "distributor_cost"] if c in df.columns), None)
    vendor_col = next((c for c in ["vendor", "supplier", "distributor"] if c in df.columns), None)
    desc_col = next((c for c in ["description", "item_description", "title", "product"] if c in df.columns), None)
    key_col = part_col or desc_col
    if key_col:
        df["_match_key"] = df[key_col].astype(str).str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
    else:
        df["_match_key"] = ""
    df["_cost_col"] = cost_col or ""
    df["_vendor_col"] = vendor_col or ""
    return df


def safe_float(v):
    try:
        if pd.isna(v):
            return None
        return float(str(v).replace("$", "").replace(",", "").strip())
    except Exception:
        return None


def price_match(query, parts, nsns, dist_df, target_margin):
    if dist_df is None or dist_df.empty:
        return None, "", None, None, "Need distributor cost"

    keys = []
    for val in [query, parts, nsns]:
        for token in str(val or "").split(","):
            k = re.sub(r"[^A-Z0-9]", "", token.upper())
            if len(k) >= 3:
                keys.append(k)
    keys = sorted(set(keys), key=len, reverse=True)

    for k in keys:
        matches = dist_df[dist_df["_match_key"].astype(str).str.contains(k, na=False, regex=False)]
        if not matches.empty:
            row = matches.iloc[0]
            cost_col = row.get("_cost_col", "")
            vendor_col = row.get("_vendor_col", "")
            cost = safe_float(row.get(cost_col)) if cost_col else None
            vendor = str(row.get(vendor_col, "")) if vendor_col else ""
            if cost:
                bid = cost / (1 - target_margin)
                return cost, vendor, bid, target_margin, "Distributor cost found"
    return None, "", None, None, "Need distributor cost"


# -----------------------------
# Scoring
# -----------------------------
def score_row(opp, attachment_text, selected_packs, mode):
    title = get_title(opp)
    desc = get_description(opp)
    notice_type = get_notice_type(opp)
    set_aside = get_set_aside(opp)
    psc = get_psc(opp)
    due = get_due_date(opp)

    summary = f"{title} {desc} {notice_type} {set_aside}"
    # Use title/desc and early attachments for product matching to avoid FAR boilerplate false positives.
    product_text = f"{summary} {attachment_text[:2500]}"
    full = f"{summary} {attachment_text}"
    full_lower = full.lower()

    hits = product_hits(product_text, selected_packs)
    broad_hits = broad_product_hits(product_text)

    if is_excluded_notice_type(notice_type):
        return 0, "Ignore", "Ignore", "Excluded notice type", hits, full

    if not is_quote_ready_notice_type(notice_type):
        return 10, "Pass", "Hard / Pass", "Not solicitation/combined solicitation", hits, full

    # Discovery lets more through; strict requires product hit and supply PSC.
    if title_obvious_pass(title):
        return 0, "Pass", "Hard / Pass", "Service/fuel/construction/software title", hits, full

    if mode == "Strict":
        if not hits:
            return 0, "Pass", "Off Category", "No selected product category match", hits, full
        if is_service_psc(psc):
            return 0, "Pass", "Hard / Pass", f"PSC {psc} appears service-based", hits, full
    elif mode == "Balanced":
        if not hits and not broad_hits:
            return 0, "Pass", "Off Category", "No product category match", hits, full
    else:  # Discovery
        # In discovery, let unknowns through if they have quote-ready/easy terms.
        pass

    easy = [x for x in EASY_TERMS if x in full_lower]
    hard = [x for x in HARD_TERMS if x in full_lower]

    hard_stop = any(x in full_lower for x in [
        "mandatory site visit", "oral presentation", "security clearance",
        "facility clearance", "construction", "renovation", "technical proposal",
        "staffing plan", "labor categories"
    ])

    if hard_stop:
        return 0, "Pass", "Hard / Pass", "Hard-stop response burden", hits, full

    score = 35
    positives = []
    risks = []

    score += 20
    positives.append("quote-ready type")

    if hits:
        score += min(30, len(hits) * 8)
        positives.append("category match")
    elif broad_hits:
        score += 10
        positives.append("broad product signal")
        risks.append("not selected category")

    set_bucket = set_aside_bucket(set_aside)
    if set_bucket == "WOSB/EDWOSB":
        score += 20
    elif set_bucket in ["Total Small Business", "Small Business"]:
        score += 10
    elif set_bucket == "No Set-Aside":
        score -= 8

    score += min(35, len(easy) * 5)
    score -= min(45, len(hard) * 8)

    dleft = days_until_due(due)
    if dleft is not None:
        if dleft < 0:
            score -= 100
            risks.append("past due")
        elif dleft <= 2:
            score -= 25
            risks.append("due very soon")
        elif dleft <= 5:
            score -= 10
        elif dleft >= 7:
            score += 8

    if attachment_text:
        score += 5

    score = max(0, min(100, score))

    if len(hard) >= 4:
        difficulty = "Medium-Hard"
    elif any(x in full_lower for x in ["authorized reseller", "authorized distributor", "brand name"]):
        difficulty = "Easy Supplier Check"
    elif len(easy) >= 5 and len(hard) <= 1:
        difficulty = "Easy Quote"
    elif len(easy) >= 3:
        difficulty = "Medium"
    else:
        difficulty = "Unknown / Review"

    if score >= 82 and difficulty in ["Easy Quote", "Easy Supplier Check", "Medium"]:
        action = "Pursue First"
    elif score >= 65:
        action = "Supplier Check"
    elif score >= 45:
        action = "Manual Review"
    else:
        action = "Pass"

    reason = f"easy={len(easy)}, hard={len(hard)}, product_hits={len(hits)}, broad_hits={len(broad_hits)}"
    return score, action, difficulty, reason, hits or broad_hits, full


# -----------------------------
# SAM API
# -----------------------------
@st.cache_data(show_spinner=False, ttl=600)
def sam_search_one(api_key, query, posted_from, posted_to, limit):
    url = "https://api.sam.gov/opportunities/v2/search"
    params = {
        "api_key": api_key,
        "q": query,
        "limit": limit,
        "postedFrom": posted_from,
        "postedTo": posted_to,
        "ptype": "o,k",
    }
    r = requests.get(url, params=params, timeout=60)
    if r.status_code != 200:
        return r.status_code, r.text, []
    try:
        data = r.json()
        return r.status_code, r.text, data.get("opportunitiesData", [])
    except Exception:
        return 500, r.text, []


# -----------------------------
# UI
# -----------------------------
with st.sidebar:
    st.header("1) Search Strategy")

    api_key = st.text_input("Paste SAM.gov API Key", type="password")

    selected_packs = st.multiselect(
        "Search Packs",
        options=list(SEARCH_PACKS.keys()),
        default=[
            "Toner / Printer / Office",
            "Janitorial / Cleaning Supplies",
            "Gloves / PPE / Safety Supplies",
            "Batteries / Power",
            "IT Peripherals",
        ],
    )

    mode = st.radio(
        "Funnel Mode",
        options=["Discovery", "Balanced", "Strict"],
        index=1,
        help="Discovery shows more maybes. Strict shows only clean product/drop-ship fits."
    )

    manual_keywords = st.text_area(
        "Extra one-per-line searches",
        value="authorized reseller\nbrand name\nfirm fixed price",
        height=90
    )

    per_query_limit = st.slider("Results per search term", 5, 50, 15)
    max_total_to_scan = st.slider("Max total unique notices to scan", 25, 300, 150)

    st.header("2) Dates / Set-Aside")

    posted_from = st.date_input("Posted From", date.today() - timedelta(days=90))
    posted_to = st.date_input("Posted To", date.today())

    set_aside_filter = st.multiselect(
        "Set-Aside",
        options=SET_ASIDE_BUCKETS,
        default=["WOSB/EDWOSB", "Total Small Business", "Small Business", "Unknown", "No Set-Aside"],
    )

    min_days_left = st.slider("Minimum Days Left", 0, 30, 1)
    max_days_left = st.slider("Maximum Days Left", 1, 180, 90)

    min_score = st.slider("Minimum Score", 0, 100, 35 if mode == "Discovery" else 45)

    hide_pass = st.checkbox("Hide Pass / Ignore", value=(mode != "Discovery"))

    st.header("3) Attachments")

    scan_attachments = st.checkbox("Scan attachment links/PDFs/ZIPs", value=True)
    scan_top_n = st.slider("Scan attachments for top N raw notices", 0, 300, 75)
    max_attachments = st.slider("Max attachments per notice", 0, 10, 3)
    max_attachment_mb = st.slider("Max MB per attachment", 1, 25, 8)
    max_pdf_pages = st.slider("Max pages per PDF", 1, 50, 15)

    st.header("4) Optional Distributor Pricing")

    distributor_file = st.file_uploader("Upload distributor pricing CSV/XLSX", type=["csv", "xlsx"])
    target_margin = st.slider("Target gross margin", 0.05, 0.50, 0.20, 0.01)


if st.button("Run Multi-Search Funnel", type="primary"):
    if not api_key:
        st.error("Paste your SAM.gov API key.")
        st.stop()

    queries = []
    for pack in selected_packs:
        queries.extend(SEARCH_PACKS.get(pack, []))
    queries.extend([x.strip() for x in manual_keywords.splitlines() if x.strip()])
    queries = list(dict.fromkeys(queries))

    if not queries:
        st.error("Select search packs or enter extra searches.")
        st.stop()

    dist_df = prepare_price_df(distributor_file)

    all_opps = {}
    status_errors = []
    progress = st.progress(0)

    st.write(f"Running {len(queries)} smaller SAM searches...")

    for i, q in enumerate(queries):
        status, raw, opps = sam_search_one(
            api_key,
            q,
            posted_from.strftime("%m/%d/%Y"),
            posted_to.strftime("%m/%d/%Y"),
            per_query_limit,
        )
        if status != 200:
            status_errors.append(f"{q}: {raw[:200]}")
        for opp in opps:
            key = get_notice_id(opp) or opp.get("noticeId") or get_ui_link(opp)
            if key:
                all_opps[key] = opp
        progress.progress((i + 1) / len(queries))

    raw_count = len(all_opps)
    opp_list = list(all_opps.values())[:max_total_to_scan]

    st.info(f"Raw unique quote-ready notices pulled: {raw_count}. Scanning/analyzing: {len(opp_list)}.")

    rows = []
    scan_progress = st.progress(0)

    for i, opp in enumerate(opp_list):
        attachment_text = ""
        attachment_status = ""
        links = get_resource_links(opp)

        if scan_attachments and i < scan_top_n:
            attachment_text, attachment_status = get_attachment_text(
                opp, api_key, max_attachments, max_attachment_mb, max_pdf_pages
            )

        score, action, difficulty, reason, hits, full_text = score_row(opp, attachment_text, selected_packs, mode)

        title = get_title(opp)
        parts = find_part_numbers(full_text)
        nsns = find_nsns(full_text)
        qtys = find_quantities(full_text)
        query = primary_query(title, hits, parts, nsns)

        cost, vendor, bid, margin, price_verdict = price_match(query, parts, nsns, dist_df, target_margin)

        set_aside = get_set_aside(opp)
        dleft = days_until_due(get_due_date(opp))

        rows.append({
            "Score": score,
            "Action": action,
            "Difficulty": difficulty,
            "Reason": reason,
            "Title": title,
            "Notice ID": get_notice_id(opp),
            "Notice Type": get_notice_type(opp),
            "Set Aside": set_aside,
            "Set Aside Bucket": set_aside_bucket(set_aside),
            "NAICS": get_naics(opp),
            "PSC": get_psc(opp),
            "Posted": get_posted_date(opp),
            "Due": get_due_date(opp),
            "Days Left": dleft,
            "Product Hits": ", ".join(hits[:8]),
            "Primary Query": query,
            "Part Numbers": parts,
            "NSNs": nsns,
            "Quantities": qtys,
            "Distributor Cost": cost,
            "Distributor Vendor": vendor,
            "Suggested Bid": bid,
            "Price Verdict": price_verdict,
            "Attachments": len(links),
            "Attachment Status": attachment_status,
            "Contact Emails": extract_emails(full_text),
            "GSA Advantage Link": gsa_advantage_link(query),
            "SAM Link": get_ui_link(opp),
            "_attachment_preview": clean_text(attachment_text[:5000]),
            "_attachment_links": links,
        })
        scan_progress.progress((i + 1) / len(opp_list))

    if not rows:
        st.warning("No results returned from SAM.")
        st.stop()

    df = pd.DataFrame(rows)

    funnel = {
        "Raw pulled": raw_count,
        "Analyzed": len(df),
        "Product hits": int((df["Product Hits"].fillna("") != "").sum()),
        "Supplier Check+": int(df["Action"].isin(["Pursue First", "Supplier Check"]).sum()),
        "Pass/Ignore": int(df["Action"].isin(["Pass", "Ignore"]).sum()),
    }

    cols = st.columns(5)
    for c, (label, val) in zip(cols, funnel.items()):
        c.metric(label, val)

    if hide_pass:
        df = df[~df["Action"].isin(["Pass", "Ignore"])]

    if set_aside_filter:
        df = df[df["Set Aside Bucket"].isin(set_aside_filter)]

    df = df[
        df["Days Left"].isna()
        | ((df["Days Left"] >= min_days_left) & (df["Days Left"] <= max_days_left))
    ]

    df = df[df["Score"] >= min_score]

    if df.empty:
        st.warning("No rows survived the display filters. Switch to Discovery mode, lower minimum score, or uncheck Hide Pass.")
        st.stop()

    df = df.sort_values(["Score", "Days Left"], ascending=[False, True], na_position="last")
    st.session_state["funnel_df"] = df

    display_cols = [
        "Score", "Action", "Difficulty", "Title", "Notice ID", "Set Aside", "NAICS", "PSC",
        "Due", "Days Left", "Product Hits", "Primary Query", "Part Numbers", "NSNs",
        "Quantities", "Price Verdict", "Distributor Cost", "Suggested Bid",
        "GSA Advantage Link", "SAM Link"
    ]

    st.success(f"Displaying {len(df)} opportunities after filters.")

    st.dataframe(
        df[display_cols],
        use_container_width=True,
        column_config={
            "SAM Link": st.column_config.LinkColumn("Open SAM"),
            "GSA Advantage Link": st.column_config.LinkColumn("Search GSA"),
            "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100),
            "Distributor Cost": st.column_config.NumberColumn("Distributor Cost", format="$%.2f"),
            "Suggested Bid": st.column_config.NumberColumn("Suggested Bid", format="$%.2f"),
        }
    )

    export = df.drop(columns=["_attachment_preview", "_attachment_links"], errors="ignore")
    st.download_button(
        "Download Funnel Results CSV",
        export.to_csv(index=False),
        "windsor_cross_sam_funnel_results.csv",
        "text/csv"
    )


if "funnel_df" in st.session_state:
    df = st.session_state["funnel_df"]
    st.markdown("---")
    st.header("Inspect One Result")

    options = [
        f"{row['Score']}/100 | {row['Action']} | {row['Notice ID']} | {row['Title'][:90]}"
        for _, row in df.iterrows()
    ]
    selected = st.selectbox("Choose result", options)

    if selected:
        notice_id = selected.split("|")[2].strip()
        row = df[df["Notice ID"] == notice_id].iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Score", int(row["Score"]))
        c2.metric("Action", row["Action"])
        c3.metric("Difficulty", row["Difficulty"])
        c4.metric("Days Left", "Unknown" if pd.isna(row["Days Left"]) else int(row["Days Left"]))

        st.markdown(f"### {row['Title']}")
        st.write(f"**Notice ID:** {row['Notice ID']}")
        st.write(f"**Set Aside:** {row['Set Aside']}")
        st.write(f"**NAICS / PSC:** {row['NAICS']} / {row['PSC']}")
        st.write(f"**Due:** {row['Due']}")
        st.write(f"**Reason:** {row['Reason']}")
        st.markdown(f"[Open in SAM]({row['SAM Link']})")
        st.markdown(f"[Search GSA Advantage]({row['GSA Advantage Link']})")

        st.subheader("Product Extraction")
        st.write(f"**Product Hits:** {row['Product Hits']}")
        st.write(f"**Primary Query:** {row['Primary Query']}")
        st.write(f"**Part Numbers:** {row['Part Numbers'] or 'None detected'}")
        st.write(f"**NSNs:** {row['NSNs'] or 'None detected'}")
        st.write(f"**Quantities:** {row['Quantities'] or 'None detected'}")

        st.subheader("Price")
        st.write(f"**Price Verdict:** {row['Price Verdict']}")
        st.write(f"**Distributor Vendor:** {row['Distributor Vendor'] or 'None matched'}")
        st.write(f"**Distributor Cost:** {row['Distributor Cost']}")
        st.write(f"**Suggested Bid:** {row['Suggested Bid']}")

        st.subheader("Attachments")
        links = row["_attachment_links"]
        if links:
            for link in links:
                st.markdown(f"- [{link['name']}]({link['url']})")
        else:
            st.info("No API attachment links found.")

        with st.expander("Attachment preview"):
            if row["_attachment_preview"]:
                st.text_area("Attachment text", row["_attachment_preview"], height=300)
            else:
                st.info("No readable attachment text scanned.")
