
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
# Windsor Cross Quote-Ready SAM Finder + Price Check - v7
#
# Purpose:
# Find EASY-TO-RESPOND, quote-ready SAM.gov product opportunities
# for a reseller/drop-ship strategy.
#
# Core workflow:
# 1. Search SAM.gov for solicitation + combined synopsis/solicitation only.
# 2. Scan available SAM attachments/PDFs/ZIPs.
# 3. Extract likely products, part numbers, NSNs, quantities, brands.
# 4. Rank by quick-hit/drop-ship fit.
# 5. Create GSA Advantage search links for market benchmarking.
# 6. Compare against uploaded distributor cost and optional GSA pricing.
# 7. Produce a simple Bid / Supplier Check / Manual Review / Pass verdict.
#
# Notes:
# - SAM often does not expose every attachment through API resourceLinks.
# - GSA Advantage does not behave like a simple public product-price API in most cases.
#   This app creates GSA Advantage search links and supports uploaded GSA price benchmarks.
# - Distributor APIs can be added later; v7 supports distributor pricing CSV upload now.
# ============================================================

st.set_page_config(page_title="Windsor Cross Quote-Ready SAM Finder v7", layout="wide")

st.title("Windsor Cross Quote-Ready SAM Finder + Price Check")
st.caption(
    "Find quote-ready product RFQs, scan attachments, extract products, benchmark GSA market pricing, "
    "and compare against distributor cost."
)


# -----------------------------
# Product Panels
# -----------------------------
PRODUCT_CATEGORY_KEYWORDS = {
    "Batteries / Power": [
        "battery", "batteries", "lithium", "power supply", "charger", "adapter", "ups"
    ],
    "Toner / Printer": [
        "toner", "cartridge", "printer", "drum", "ink", "mfd", "copier"
    ],
    "Gloves / PPE": [
        "gloves", "ppe", "nitrile", "latex", "safety glasses", "mask", "respirator", "protective"
    ],
    "Janitorial / Cleaning": [
        "janitorial", "cleaning", "disinfectant", "paper towels", "trash bags", "wipes",
        "detergent", "custodial supplies", "soap", "sanitizer"
    ],
    "Office Supplies": [
        "office supplies", "copy paper", "paper", "folders", "pens", "staples", "binders",
        "envelopes", "labels"
    ],
    "IT Peripherals": [
        "monitor", "keyboard", "mouse", "dock", "cable", "adapter", "headset", "webcam",
        "scanner", "external drive"
    ],
    "Safety / Industrial Supplies": [
        "safety", "first aid", "hard hat", "earplug", "helmet", "cones", "vest", "eyewash",
        "spill kit"
    ],
    "Uniforms / Apparel": [
        "uniform", "apparel", "shirt", "pants", "boots", "coveralls", "jacket"
    ],
}

DEFAULT_CATEGORIES = [
    "Batteries / Power",
    "Toner / Printer",
    "Gloves / PPE",
    "Janitorial / Cleaning",
    "Office Supplies",
    "Safety / Industrial Supplies",
]


# -----------------------------
# Scoring Terms
# -----------------------------
EASY_TERMS = {
    "rfq": 18,
    "request for quote": 18,
    "quote": 10,
    "quotation": 10,
    "firm fixed price": 16,
    "ffp": 12,
    "fob destination": 12,
    "commercial product": 12,
    "commercial item": 12,
    "cots": 12,
    "brand name": 14,
    "brand name or equal": 14,
    "authorized reseller": 18,
    "authorized distributor": 18,
    "distributor": 8,
    "supplier": 6,
    "unit price": 8,
    "total price": 8,
    "delivery": 8,
    "lead time": 8,
    "quantity": 8,
    "purchase order": 10,
    "simplified acquisition": 12,
    "lowest price": 12,
    "lpta": 10,
    "email": 5,
}

SET_ASIDE_TERMS = {
    "women-owned": 20,
    "wosb": 20,
    "edwosb": 20,
    "economically disadvantaged women-owned": 20,
    "total small business": 12,
    "small business set-aside": 12,
    "small business": 6,
}

HARD_TERMS = {
    "award notice": -100,
    "sources sought": -80,
    "presolicitation": -60,
    "pre-solicitation": -60,
    "oral presentation": -45,
    "in-person presentation": -45,
    "site visit": -35,
    "mandatory site visit": -50,
    "bafo": -30,
    "best and final": -30,
    "technical proposal": -40,
    "technical volume": -40,
    "management approach": -30,
    "quality control plan": -25,
    "staffing plan": -40,
    "key personnel": -45,
    "resume": -25,
    "past performance": -25,
    "cpars": -20,
    "security clearance": -60,
    "facility clearance": -60,
    "secret clearance": -60,
    "statement of work": -20,
    "performance work statement": -25,
    "pws": -20,
    "labor category": -40,
    "labor categories": -40,
    "staffing": -40,
    "onsite": -30,
    "on-site": -30,
    "installation": -30,
    "maintenance services": -30,
    "service technician": -35,
    "subcontracting plan": -35,
    "bond": -40,
    "bid guarantee": -40,
    "construction": -60,
    "renovation": -60,
    "repair": -35,
    "calibration": -45,
    "tmde": -45,
    "service contract act": -35,
    "wage determination": -30,
}

REQUIREMENT_PATTERNS = {
    "Email quote likely": r"(email.*quote|quote.*email|submit.*quote|send.*quote)",
    "CAGE/UEI likely needed": r"\b(cage|uei|sam registration|unique entity)\b",
    "Delivery lead time requested": r"\b(lead time|delivery schedule|days aro|after receipt of order)\b",
    "Authorized reseller proof": r"\b(authorized reseller|authorized distributor|authorization letter|reseller letter)\b",
    "Brand name": r"\b(brand name|brand-name)\b",
    "Unit/total price": r"\b(unit price|total price|extended price)\b",
    "SF 1449/FAR clauses": r"\b(sf\s*1449|far|dfars)\b",
    "Technical narrative": r"\b(technical proposal|technical volume|technical approach)\b",
    "Past performance": r"\b(past performance|cpars|references)\b",
    "Site visit": r"\b(site visit|walk.?through)\b",
    "Oral presentation": r"\b(oral presentation|presentation)\b",
}


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


def notice_is_quote_ready_type(notice_type):
    n = notice_type.lower()
    return ("solicitation" in n or "combined" in n) and "award" not in n and "sources sought" not in n and "pre" not in n


def notice_is_excluded_type(notice_type):
    n = notice_type.lower()
    return "award" in n or "sources sought" in n or "pre-solicitation" in n or "presolicitation" in n


def category_terms(selected_categories):
    terms = []
    for cat in selected_categories:
        terms.extend(PRODUCT_CATEGORY_KEYWORDS.get(cat, []))
    return sorted(set([t.lower() for t in terms]))


def category_hits(text, selected_categories):
    t = text.lower()
    hits = []
    for term in category_terms(selected_categories):
        if re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", t):
            hits.append(term)
    return hits


def build_keyword_from_categories(selected_categories):
    terms = category_terms(selected_categories)
    return " OR ".join(terms[:35])


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


def parse_money(value):
    if value in (None, "", [], {}):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    matches = re.findall(r"[\$]?\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)", str(value))
    if not matches:
        return None
    try:
        return float(matches[0].replace(",", ""))
    except Exception:
        return None


def get_value(opp):
    candidates = [
        opp.get("awardAmount"),
        opp.get("awardValue"),
        opp.get("estimatedValue"),
        opp.get("baseAndAllOptionsValue"),
    ]
    award = opp.get("award")
    if isinstance(award, dict):
        candidates.extend([award.get("amount"), award.get("awardAmount"), award.get("awardValue")])
    for value in candidates:
        parsed = parse_money(value)
        if parsed is not None:
            return parsed
    return None


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


def extract_emails(text):
    emails = sorted(set(re.findall(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", text or "")))
    return ", ".join(emails[:5])


def matched_requirements(text):
    found = []
    for label, pattern in REQUIREMENT_PATTERNS.items():
        if re.search(pattern, text, flags=re.IGNORECASE):
            found.append(label)
    return "; ".join(found)


def gsa_advantage_link(query):
    query = clean_text(query)
    if not query:
        return ""
    return f"https://www.gsaadvantage.gov/advantage/ws/search/advantage_search?keyword={quote_plus(query)}"


# -----------------------------
# Attachment extraction
# -----------------------------
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
            for row in ws.iter_rows(max_row=80, values_only=True):
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
            for name in z.namelist()[:25]:
                try:
                    data = z.read(name)
                except Exception:
                    continue
                lower = name.lower()
                chunks.append(f"\n--- FILE: {name} ---\n")
                if lower.endswith(".pdf") or data[:4] == b"%PDF":
                    chunks.append(extract_pdf_text(data, max_pages=max_pages))
                elif lower.endswith(".docx"):
                    chunks.append(extract_docx_text(data))
                elif lower.endswith(".xlsx"):
                    chunks.append(extract_xlsx_text(data))
                elif lower.endswith((".txt", ".csv")):
                    chunks.append(data.decode("utf-8", errors="ignore"))
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
        for params in [{}, {"api_key": api_key}]:
            r = requests.get(
                url,
                params=params,
                timeout=45,
                headers={"User-Agent": "WindsorCrossQuoteReady/1.0"},
            )
            if r.status_code != 200:
                continue

            if len(r.content) > max_mb * 1024 * 1024:
                return "", f"Skipped > {max_mb}MB"

            content = r.content
            content_type = r.headers.get("content-type", "").lower()
            lower = url.lower()

            if content[:4] == b"%PDF" or ".pdf" in lower or "pdf" in content_type:
                txt = extract_pdf_text(content, max_pages=max_pages)
                return txt, "PDF scanned" if txt else "PDF downloaded; no text extracted"

            if content[:2] == b"PK" or ".zip" in lower or "zip" in content_type:
                txt = extract_zip_text(content, max_pages=max_pages)
                return txt, "ZIP/Office scanned" if txt else "ZIP/Office downloaded; no text extracted"

            if ".docx" in lower:
                return extract_docx_text(content), "DOCX scanned"

            if ".xlsx" in lower:
                return extract_xlsx_text(content), "XLSX scanned"

            if "text" in content_type or "json" in content_type or "xml" in content_type:
                return r.text, "Text scanned"

        return "", "Could not download readable attachment"
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


# -----------------------------
# Product extraction
# -----------------------------
def find_nsns(text):
    # NSN is typically 13 digits, often formatted 1234-01-234-5678.
    patterns = [
        r"\b\d{4}[-\s]?\d{2}[-\s]?\d{3}[-\s]?\d{4}\b",
        r"\b\d{13}\b",
    ]
    found = []
    for pattern in patterns:
        found.extend(re.findall(pattern, text))
    return sorted(set([x.strip() for x in found]))


def find_quantities(text):
    # Pull likely quantities from common RFQ language.
    patterns = [
        r"\bqty\.?\s*[:\-]?\s*(\d{1,6})\b",
        r"\bquantity\s*[:\-]?\s*(\d{1,6})\b",
        r"\b(\d{1,6})\s*(each|ea|boxes|box|units|unit|cases|case|pairs|pair)\b",
    ]
    found = []
    for pattern in patterns:
        for m in re.findall(pattern, text, flags=re.IGNORECASE):
            if isinstance(m, tuple):
                found.append(" ".join([str(x) for x in m if x]))
            else:
                found.append(str(m))
    return sorted(set(found))


def find_part_numbers(text):
    # Conservative part number extraction. Avoids generic dates and clause numbers where possible.
    patterns = [
        r"\bpart(?:\s+number|\s+no\.?|#)?\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-_/\.]{2,30})\b",
        r"\bp/n\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-_/\.]{2,30})\b",
        r"\bmodel\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-_/\.]{2,30})\b",
        r"\bmanufacturer part number\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-_/\.]{2,30})\b",
        r"\bmpn\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-_/\.]{2,30})\b",
    ]
    found = []
    upper = text.upper()
    for pattern in patterns:
        found.extend(re.findall(pattern, upper, flags=re.IGNORECASE))

    # Clean obvious false positives
    bad = {"THE", "AND", "FOR", "WITH", "FROM", "THIS", "THAT", "QUOTE", "SOLICITATION"}
    cleaned = []
    for x in found:
        x = x.strip(" .,:;()[]{}")
        if x and x not in bad and not re.fullmatch(r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}", x):
            cleaned.append(x)
    return sorted(set(cleaned))[:8]


def find_brand_names(text):
    # Extract brands/manufacturers from common language.
    patterns = [
        r"\bbrand name\s*[:\-]?\s*([A-Z][A-Za-z0-9&\-\s]{2,40})",
        r"\bmanufacturer\s*[:\-]?\s*([A-Z][A-Za-z0-9&\-\s]{2,40})",
        r"\bmfr\.?\s*[:\-]?\s*([A-Z][A-Za-z0-9&\-\s]{2,40})",
        r"\bmake\s*[:\-]?\s*([A-Z][A-Za-z0-9&\-\s]{2,40})",
    ]
    found = []
    for pattern in patterns:
        for m in re.findall(pattern, text, flags=re.IGNORECASE):
            m = clean_text(m)
            # stop at common boundary words
            m = re.split(r"\b(model|part|p/n|qty|quantity|shall|must|with|for)\b", m, flags=re.IGNORECASE)[0]
            found.append(clean_text(m))
    return sorted(set([x for x in found if len(x) >= 2]))[:5]


def extract_product_summary(title, full_text, selected_categories):
    cat_hits = category_hits(full_text, selected_categories)
    nsns = find_nsns(full_text)
    parts = find_part_numbers(full_text)
    qtys = find_quantities(full_text)
    brands = find_brand_names(full_text)

    # Choose best query for GSA/distributor matching.
    if parts:
        primary_query = parts[0]
    elif nsns:
        primary_query = nsns[0]
    elif brands and cat_hits:
        primary_query = f"{brands[0]} {cat_hits[0]}"
    elif cat_hits:
        primary_query = cat_hits[0]
    else:
        primary_query = title

    return {
        "Product Match Terms": ", ".join(cat_hits[:8]),
        "Primary Product Query": clean_text(primary_query),
        "Part Numbers": ", ".join(parts),
        "NSNs": ", ".join(nsns),
        "Quantities Found": ", ".join(qtys[:8]),
        "Brands/Manufacturers": ", ".join(brands),
        "GSA Advantage Link": gsa_advantage_link(primary_query),
    }


# -----------------------------
# Pricing match
# -----------------------------
def normalize_key(value):
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def prepare_price_df(uploaded_file, expected_name):
    if uploaded_file is None:
        return pd.DataFrame()

    try:
        if uploaded_file.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.warning(f"Could not read {expected_name}: {e}")
        return pd.DataFrame()

    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    # Accept a few common column names.
    part_cols = ["part_number", "part", "sku", "mpn", "manufacturer_part_number", "nsn"]
    desc_cols = ["description", "item_description", "title", "product"]
    cost_cols = ["unit_cost", "cost", "price", "your_cost", "distributor_cost"]
    low_cols = ["gsa_low", "low_price", "lowest_price", "gsa_low_price"]
    avg_cols = ["gsa_avg", "average_price", "avg_price", "gsa_average_price"]
    vendor_cols = ["vendor", "supplier", "distributor"]

    def pick(cols):
        for c in cols:
            if c in df.columns:
                return c
        return None

    part_col = pick(part_cols)
    desc_col = pick(desc_cols)
    cost_col = pick(cost_cols)
    low_col = pick(low_cols)
    avg_col = pick(avg_cols)
    vendor_col = pick(vendor_cols)

    if part_col:
        df["_match_key"] = df[part_col].apply(normalize_key)
    elif desc_col:
        df["_match_key"] = df[desc_col].apply(normalize_key)
    else:
        df["_match_key"] = ""

    df["_part_col"] = part_col or ""
    df["_desc_col"] = desc_col or ""
    df["_cost_col"] = cost_col or ""
    df["_low_col"] = low_col or ""
    df["_avg_col"] = avg_col or ""
    df["_vendor_col"] = vendor_col or ""

    return df


def find_price_match(product_query, part_numbers, nsns, price_df):
    if price_df is None or price_df.empty:
        return None

    keys = []
    for source in [product_query, part_numbers, nsns]:
        for token in re.split(r"[,;\n]", str(source or "")):
            k = normalize_key(token)
            if len(k) >= 3:
                keys.append(k)

    keys = sorted(set(keys), key=len, reverse=True)

    if not keys:
        return None

    for key in keys:
        matches = price_df[price_df["_match_key"].astype(str).str.contains(key, na=False, regex=False)]
        if not matches.empty:
            return matches.iloc[0].to_dict()

    return None


def safe_float(value):
    try:
        if pd.isna(value):
            return None
        return float(str(value).replace("$", "").replace(",", "").strip())
    except Exception:
        return None


def price_analysis(product_query, part_numbers, nsns, distributor_df, gsa_df, target_margin, shipping_per_unit):
    dist = find_price_match(product_query, part_numbers, nsns, distributor_df)
    gsa = find_price_match(product_query, part_numbers, nsns, gsa_df)

    dist_cost = None
    dist_vendor = ""
    dist_note = ""

    if dist:
        cost_col = dist.get("_cost_col", "")
        vendor_col = dist.get("_vendor_col", "")
        dist_cost = safe_float(dist.get(cost_col)) if cost_col else None
        dist_vendor = str(dist.get(vendor_col, "")) if vendor_col else ""
        dist_note = "Distributor match found"

    gsa_low = None
    gsa_avg = None
    gsa_note = ""

    if gsa:
        low_col = gsa.get("_low_col", "")
        avg_col = gsa.get("_avg_col", "")
        cost_col = gsa.get("_cost_col", "")

        # If user uploads one price column as "price", treat it as low if no explicit gsa_low.
        gsa_low = safe_float(gsa.get(low_col)) if low_col else None
        gsa_avg = safe_float(gsa.get(avg_col)) if avg_col else None

        if gsa_low is None and cost_col:
            gsa_low = safe_float(gsa.get(cost_col))

        gsa_note = "GSA benchmark match found"

    landed_cost = None
    suggested_bid = None
    gross_margin_pct = None
    price_verdict = "Need pricing"

    if dist_cost is not None:
        landed_cost = dist_cost + float(shipping_per_unit or 0)
        if target_margin >= 1:
            target_margin = target_margin / 100
        target_price = landed_cost / max(0.01, (1 - target_margin))

        if gsa_avg:
            # Try to stay slightly under average GSA market where possible.
            suggested_bid = min(target_price, gsa_avg * 0.98)
        elif gsa_low:
            # If only low GSA is known, target slightly above cost but not wildly above market.
            suggested_bid = min(target_price, gsa_low * 1.05)
        else:
            suggested_bid = target_price

        if suggested_bid > 0:
            gross_margin_pct = (suggested_bid - landed_cost) / suggested_bid

        if gross_margin_pct is not None:
            if gross_margin_pct >= target_margin and (gsa_low is None or suggested_bid <= gsa_low * 1.15):
                price_verdict = "Price looks workable"
            elif gross_margin_pct >= 0.10:
                price_verdict = "Thin margin / review"
            else:
                price_verdict = "Margin too thin"
    else:
        if gsa_low or gsa_avg:
            price_verdict = "Need distributor cost"
        else:
            price_verdict = "Use GSA link / supplier call"

    return {
        "Distributor Match": dist_note,
        "Distributor Vendor": dist_vendor,
        "Distributor Unit Cost": dist_cost,
        "GSA Match": gsa_note,
        "GSA Low": gsa_low,
        "GSA Avg": gsa_avg,
        "Landed Unit Cost": landed_cost,
        "Suggested Unit Bid": suggested_bid,
        "Estimated Gross Margin": gross_margin_pct,
        "Price Verdict": price_verdict,
    }


# -----------------------------
# Difficulty / scoring
# -----------------------------
def difficulty_from_text(full_text, notice_type, cat_hits):
    t = full_text.lower()

    if notice_is_excluded_type(notice_type):
        return "Ignore", "Not a quote-ready solicitation type"

    hard_hits = [term for term in HARD_TERMS if term in t]
    easy_hits = [term for term in EASY_TERMS if term in t]

    hard_stops = [
        "mandatory site visit",
        "oral presentation",
        "security clearance",
        "facility clearance",
        "construction",
        "renovation",
        "technical proposal",
        "staffing plan",
        "labor categories",
    ]

    if any(x in t for x in hard_stops):
        return "Hard / Pass", "Hard-stop response burden detected"

    if not cat_hits:
        return "Off Category", "Does not match selected Windsor Cross product categories"

    if any(x in t for x in ["authorized reseller", "authorized distributor", "brand name"]) and len(easy_hits) >= 3 and len(hard_hits) <= 1:
        return "Easy Supplier Check", "Simple product RFQ; supplier pricing/authorization is main task"

    if len(easy_hits) >= 6 and len(hard_hits) <= 1:
        return "Easy Quote", "Likely straightforward quote response"

    if len(easy_hits) >= 4 and len(hard_hits) <= 2:
        return "Medium", "Likely manageable; check attachments"

    if len(hard_hits) >= 3:
        return "Medium-Hard", "Several proposal burden terms detected"

    return "Unknown / Review", "Not enough response detail found"


def score_opp(opp, attachment_text, selected_categories, include_terms, exclude_terms):
    title = clean_text(opp.get("title", ""))
    desc = clean_text(opp.get("description", ""))
    notice_type = get_notice_type(opp)
    set_aside = get_set_aside(opp)
    due = get_due_date(opp)

    summary = f"{title} {desc} {notice_type} {set_aside}"
    full_text = f"{summary} {attachment_text}"
    t = full_text.lower()

    score = 35
    positives = []
    risks = []

    # Hard filter by notice type
    if notice_is_excluded_type(notice_type):
        return {
            "score": 0,
            "action": "Ignore",
            "difficulty": "Ignore",
            "reason": "Excluded notice type: award/sources sought/pre-solicitation",
            "positive": "",
            "risks": "Not a quote-ready solicitation",
            "requirements": "",
            "emails": "",
            "days_left": days_until_due(due),
            "full_text": full_text,
        }

    if notice_is_quote_ready_type(notice_type):
        score += 20
        positives.append("Quote-ready notice type")
    else:
        score -= 20
        risks.append("Notice type not clearly quote-ready")

    cat_hits = category_hits(full_text, selected_categories)
    if cat_hits:
        score += min(35, len(cat_hits) * 8)
        positives.append("Product match: " + ", ".join(cat_hits[:6]))
    else:
        return {
            "score": 0,
            "action": "Pass",
            "difficulty": "Off Category",
            "reason": "Does not match selected product categories",
            "positive": "",
            "risks": "Off-category product/service",
            "requirements": matched_requirements(full_text),
            "emails": extract_emails(full_text),
            "days_left": days_until_due(due),
            "full_text": full_text,
        }

    for term, pts in SET_ASIDE_TERMS.items():
        if term in t:
            score += pts
            positives.append(term)

    if "no set aside" in set_aside.lower():
        score -= 12
        risks.append("No set-aside")

    for term, pts in EASY_TERMS.items():
        if term in t:
            score += pts
            positives.append(term)

    for term, pts in HARD_TERMS.items():
        if term in t:
            score += pts
            risks.append(term)

    include_list = [x.strip().lower() for x in re.split(r"[,;\n]", include_terms or "") if x.strip()]
    exclude_list = [x.strip().lower() for x in re.split(r"[,;\n]", exclude_terms or "") if x.strip()]

    include_hits = [x for x in include_list if x in t]
    exclude_hits = [x for x in exclude_list if x in t]

    if include_hits:
        score += min(20, len(include_hits) * 5)
        positives.append("Include hits: " + ", ".join(include_hits[:5]))

    if exclude_hits:
        score -= min(70, len(exclude_hits) * 18)
        risks.append("Exclude hits: " + ", ".join(exclude_hits[:5]))

    dleft = days_until_due(due)
    if dleft is not None:
        if dleft < 0:
            score -= 100
            risks.append("Past due")
        elif dleft <= 2:
            score -= 35
            risks.append("Due in 0-2 days")
        elif dleft <= 5:
            score -= 12
            risks.append("Due soon")
        elif dleft >= 7:
            score += 8
            positives.append("Enough time to quote")

    if attachment_text:
        positives.append("Attachments scanned")
    else:
        risks.append("No attachment text scanned")

    difficulty, reason = difficulty_from_text(full_text, notice_type, cat_hits)

    if difficulty == "Easy Quote":
        score += 15
    elif difficulty == "Easy Supplier Check":
        score += 12
    elif difficulty == "Medium-Hard":
        score -= 20
    elif difficulty == "Hard / Pass":
        score -= 50

    score = max(0, min(100, score))

    if difficulty in ["Hard / Pass", "Off Category"] or score < 45:
        action = "Pass"
    elif score >= 85 and difficulty in ["Easy Quote", "Easy Supplier Check", "Medium"]:
        action = "Pursue First"
    elif score >= 70:
        action = "Supplier Check"
    elif score >= 55:
        action = "Manual Review"
    else:
        action = "Pass"

    return {
        "score": score,
        "action": action,
        "difficulty": difficulty,
        "reason": reason,
        "positive": "; ".join(dict.fromkeys(positives[:10])),
        "risks": "; ".join(dict.fromkeys(risks[:10])),
        "requirements": matched_requirements(full_text),
        "emails": extract_emails(full_text),
        "days_left": dleft,
        "full_text": full_text,
    }


# -----------------------------
# SAM API
# -----------------------------
@st.cache_data(show_spinner=False, ttl=600)
def sam_search(api_key, keyword, posted_from, posted_to, limit):
    url = "https://api.sam.gov/opportunities/v2/search"
    params = {
        "api_key": api_key,
        "q": keyword,
        "limit": limit,
        "postedFrom": posted_from,
        "postedTo": posted_to,
        # Quote-ready only:
        # o = Solicitation
        # k = Combined Synopsis/Solicitation
        "ptype": "o,k",
    }
    r = requests.get(url, params=params, timeout=60)
    try:
        data = r.json() if r.status_code == 200 else None
    except Exception:
        data = None
    return r.status_code, r.text, data


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("1) Windsor Cross Product Fit")

    selected_categories = st.multiselect(
        "Product Categories",
        options=list(PRODUCT_CATEGORY_KEYWORDS.keys()),
        default=DEFAULT_CATEGORIES,
        help="Only opportunities matching these product categories survive."
    )

    search_from_categories = st.checkbox(
        "Build SAM search from selected categories",
        value=True,
        help="Recommended. Keeps the search aligned to what Windsor Cross can resell."
    )

    manual_keywords = st.text_area(
        "Optional extra keywords",
        value="",
        height=70,
        placeholder="Example: authorized reseller OR brand name"
    )

    base_keyword = build_keyword_from_categories(selected_categories) if search_from_categories else ""
    if manual_keywords.strip() and base_keyword:
        keyword = f"({base_keyword}) OR ({manual_keywords.strip()})"
    elif manual_keywords.strip():
        keyword = manual_keywords.strip()
    else:
        keyword = base_keyword

    st.caption("Actual SAM keyword search:")
    st.code(keyword or "(blank)", language="text")

    st.header("2) SAM Search Window")

    api_key = st.text_input("Paste SAM.gov API Key", type="password")

    col1, col2 = st.columns(2)
    with col1:
        posted_from = st.date_input("Posted From", date.today() - timedelta(days=30))
    with col2:
        posted_to = st.date_input("Posted To", date.today())

    limit = st.slider("SAM Results to Review", 10, 100, 50)

    st.info("Notice types are locked to Solicitation + Combined Synopsis/Solicitation.")

    st.header("3) Set-Aside / Timing")

    set_aside_filter = st.multiselect(
        "Set-Aside",
        options=["WOSB/EDWOSB", "Total Small Business", "Small Business", "No Set-Aside", "Unknown"],
        default=["WOSB/EDWOSB", "Total Small Business", "Small Business", "Unknown"],
    )

    min_score = st.slider("Minimum Score", 0, 100, 55)
    min_days_left = st.slider("Minimum Days Left", 0, 30, 3)
    max_days_left = st.slider("Maximum Days Left", 1, 90, 45)

    hide_pass = st.checkbox("Hide Pass / Ignore", value=True)

    st.header("4) Response Burden")

    include_terms = st.text_area(
        "Boost terms",
        value="authorized reseller\nbrand name\nfirm fixed price\nfob destination\nquote\nlowest price\nunit price\nlead time",
        height=110
    )

    exclude_terms = st.text_area(
        "Auto-penalty terms",
        value="site visit\noral presentation\nbafo\ntechnical proposal\nconstruction\nrenovation\nstaffing\nsecurity clearance\nfacility clearance\ncalibration\ninstallation",
        height=130
    )

    st.header("5) Attachment Scan")

    scan_attachments = st.checkbox("Scan SAM attachment links/PDFs/ZIPs", value=True)
    scan_top_n = st.slider("Scan attachments for top N raw results", 0, 100, 35)
    max_attachments = st.slider("Max attachments per opportunity", 0, 10, 4)
    max_attachment_mb = st.slider("Max MB per attachment", 1, 25, 8)
    max_pdf_pages = st.slider("Max pages per PDF", 1, 50, 20)

    st.header("6) Price Check")

    distributor_file = st.file_uploader(
        "Upload distributor pricing CSV/XLSX",
        type=["csv", "xlsx"],
        help="Columns can include: part_number, sku, nsn, description, unit_cost, vendor, stock"
    )

    gsa_file = st.file_uploader(
        "Upload GSA benchmark CSV/XLSX",
        type=["csv", "xlsx"],
        help="Columns can include: part_number, nsn, description, gsa_low, gsa_avg, price"
    )

    target_margin = st.slider("Target gross margin", 0.05, 0.50, 0.20, 0.01)
    shipping_per_unit = st.number_input("Estimated shipping/handling per unit", min_value=0.0, value=0.0, step=1.0)


# -----------------------------
# Main
# -----------------------------
if st.button("Find Quote-Ready Opportunities + Price Check", type="primary"):
    if not api_key:
        st.error("Please paste your SAM.gov API key.")
        st.stop()

    if not keyword.strip():
        st.error("Select at least one product category or enter keywords.")
        st.stop()

    if posted_from > posted_to:
        st.error("Posted From cannot be after Posted To.")
        st.stop()

    distributor_df = prepare_price_df(distributor_file, "distributor pricing")
    gsa_df = prepare_price_df(gsa_file, "GSA benchmark pricing")

    with st.spinner("Searching quote-ready SAM notices..."):
        status, raw, data = sam_search(
            api_key=api_key,
            keyword=keyword,
            posted_from=posted_from.strftime("%m/%d/%Y"),
            posted_to=posted_to.strftime("%m/%d/%Y"),
            limit=limit,
        )

    if status != 200:
        st.error("SAM.gov API error")
        st.text(raw)
        st.stop()

    opportunities = data.get("opportunitiesData", [])

    if not opportunities:
        st.warning("No SAM opportunities returned.")
        st.json(data)
        st.stop()

    rows = []
    progress = st.progress(0)

    for i, opp in enumerate(opportunities):
        title = clean_text(opp.get("title", ""))
        attachment_text = ""
        attachment_status = []
        links = get_resource_links(opp)

        if scan_attachments and i < scan_top_n:
            attachment_text, attachment_status = get_attachment_text(
                opp,
                api_key=api_key,
                max_attachments=max_attachments,
                max_mb=max_attachment_mb,
                max_pages=max_pdf_pages,
            )

        result = score_opp(
            opp=opp,
            attachment_text=attachment_text,
            selected_categories=selected_categories,
            include_terms=include_terms,
            exclude_terms=exclude_terms,
        )

        full_text = result.get("full_text", "")
        product = extract_product_summary(title, full_text, selected_categories)

        price = price_analysis(
            product_query=product["Primary Product Query"],
            part_numbers=product["Part Numbers"],
            nsns=product["NSNs"],
            distributor_df=distributor_df,
            gsa_df=gsa_df,
            target_margin=target_margin,
            shipping_per_unit=shipping_per_unit,
        )

        set_aside = get_set_aside(opp)

        # Final action refinement based on pricing.
        final_action = result["action"]
        if final_action in ["Pursue First", "Supplier Check"]:
            if price["Price Verdict"] == "Margin too thin":
                final_action = "Pass - Price"
            elif price["Price Verdict"] == "Need distributor cost":
                final_action = "Supplier Check"
            elif price["Price Verdict"] == "Use GSA link / supplier call":
                final_action = "Supplier Check"

        rows.append({
            "Score": result["score"],
            "Action": final_action,
            "Difficulty": result["difficulty"],
            "Why Difficulty": result["reason"],
            "Title": title,
            "Notice ID": get_notice_id(opp),
            "Notice Type": get_notice_type(opp),
            "Set Aside": set_aside,
            "Set Aside Bucket": set_aside_bucket(set_aside),
            "NAICS": get_naics(opp),
            "PSC": get_psc(opp),
            "Posted": get_posted_date(opp),
            "Due": get_due_date(opp),
            "Days Left": result["days_left"],
            "Value": get_value(opp),
            "Product Match Terms": product["Product Match Terms"],
            "Primary Product Query": product["Primary Product Query"],
            "Part Numbers": product["Part Numbers"],
            "NSNs": product["NSNs"],
            "Quantities Found": product["Quantities Found"],
            "Brands/Manufacturers": product["Brands/Manufacturers"],
            "GSA Advantage Link": product["GSA Advantage Link"],
            "Distributor Unit Cost": price["Distributor Unit Cost"],
            "Distributor Vendor": price["Distributor Vendor"],
            "GSA Low": price["GSA Low"],
            "GSA Avg": price["GSA Avg"],
            "Landed Unit Cost": price["Landed Unit Cost"],
            "Suggested Unit Bid": price["Suggested Unit Bid"],
            "Estimated Gross Margin": price["Estimated Gross Margin"],
            "Price Verdict": price["Price Verdict"],
            "Attachments": len(links),
            "Response Requirements": result["requirements"],
            "Contact Emails": result["emails"],
            "Positive Signals": result["positive"],
            "Pass/Risk Reasons": result["risks"],
            "SAM Link": get_ui_link(opp),
            "Attachment Status": "; ".join(attachment_status[:5]),
            "_attachment_preview": clean_text(attachment_text[:5000]),
            "_attachment_links": links,
        })

        progress.progress((i + 1) / len(opportunities))

    df = pd.DataFrame(rows)

    if hide_pass:
        df = df[~df["Action"].isin(["Pass", "Ignore", "Pass - Price"])]

    df = df[df["Score"] >= min_score]

    if set_aside_filter:
        df = df[df["Set Aside Bucket"].isin(set_aside_filter)]

    df = df[
        df["Days Left"].isna()
        | ((df["Days Left"] >= min_days_left) & (df["Days Left"] <= max_days_left))
    ]

    if df.empty:
        st.warning("No quote-ready opportunities survived. That is okay — the filters are doing their job.")
        st.stop()

    df = df.sort_values(by=["Score", "Days Left"], ascending=[False, True], na_position="last")
    st.session_state["quote_ready_df"] = df

    st.success(f"Found {len(df)} quote-ready opportunities after filters")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Pursue First", int((df["Action"] == "Pursue First").sum()))
    c2.metric("Supplier Check", int((df["Action"] == "Supplier Check").sum()))
    c3.metric("Manual Review", int((df["Action"] == "Manual Review").sum()))
    c4.metric("Easy Quote", int((df["Difficulty"] == "Easy Quote").sum()))
    c5.metric("Easy Supplier Check", int((df["Difficulty"] == "Easy Supplier Check").sum()))

    display_cols = [
        "Score", "Action", "Difficulty", "Price Verdict", "Product Match Terms",
        "Primary Product Query", "Part Numbers", "NSNs", "Quantities Found",
        "Title", "Notice ID", "Set Aside", "NAICS", "PSC", "Due", "Days Left",
        "Distributor Unit Cost", "GSA Low", "GSA Avg", "Suggested Unit Bid",
        "Estimated Gross Margin", "GSA Advantage Link", "SAM Link"
    ]

    st.dataframe(
        df[display_cols],
        use_container_width=True,
        column_config={
            "SAM Link": st.column_config.LinkColumn("Open in SAM"),
            "GSA Advantage Link": st.column_config.LinkColumn("Search GSA Advantage"),
            "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100),
            "Distributor Unit Cost": st.column_config.NumberColumn("Distributor Unit Cost", format="$%.2f"),
            "GSA Low": st.column_config.NumberColumn("GSA Low", format="$%.2f"),
            "GSA Avg": st.column_config.NumberColumn("GSA Avg", format="$%.2f"),
            "Suggested Unit Bid": st.column_config.NumberColumn("Suggested Unit Bid", format="$%.2f"),
            "Estimated Gross Margin": st.column_config.NumberColumn("Estimated Gross Margin", format="%.1%"),
        }
    )

    export = df.drop(columns=["_attachment_preview", "_attachment_links"], errors="ignore")
    st.download_button(
        "Download Quote-Ready CSV",
        export.to_csv(index=False),
        "windsor_cross_quote_ready_price_checked_results.csv",
        "text/csv"
    )


# -----------------------------
# Review panel
# -----------------------------
if "quote_ready_df" in st.session_state:
    df = st.session_state["quote_ready_df"]

    st.markdown("---")
    st.header("Inspect One Opportunity")

    options = [
        f"{row['Score']}/100 | {row['Action']} | {row['Notice ID']} | {row['Primary Product Query']} | {row['Title'][:75]}"
        for _, row in df.iterrows()
    ]

    selected = st.selectbox("Choose an opportunity", options)

    if selected:
        # Extract notice id from option
        parts = selected.split("|")
        notice_id = parts[2].strip()
        row = df[df["Notice ID"] == notice_id].iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Score", int(row["Score"]))
        c2.metric("Action", row["Action"])
        c3.metric("Difficulty", row["Difficulty"])
        c4.metric("Price Verdict", row["Price Verdict"])

        st.markdown(f"### {row['Title']}")
        st.write(f"**Notice ID:** {row['Notice ID']}")
        st.write(f"**Notice Type:** {row['Notice Type']}")
        st.write(f"**Set Aside:** {row['Set Aside']}")
        st.write(f"**NAICS / PSC:** {row['NAICS']} / {row['PSC']}")
        st.write(f"**Due:** {row['Due']}")
        st.markdown(f"[Open in SAM]({row['SAM Link']})")

        st.subheader("Extracted Product")
        st.write(f"**Primary Product Query:** {row['Primary Product Query']}")
        st.write(f"**Product Match Terms:** {row['Product Match Terms']}")
        st.write(f"**Part Numbers:** {row['Part Numbers'] or 'None detected'}")
        st.write(f"**NSNs:** {row['NSNs'] or 'None detected'}")
        st.write(f"**Quantities Found:** {row['Quantities Found'] or 'None detected'}")
        st.write(f"**Brands/Manufacturers:** {row['Brands/Manufacturers'] or 'None detected'}")
        st.markdown(f"[Search GSA Advantage]({row['GSA Advantage Link']})")

        st.subheader("Price Check")
        price_cols = st.columns(4)
        price_cols[0].metric("Distributor Cost", "N/A" if pd.isna(row["Distributor Unit Cost"]) else f"${row['Distributor Unit Cost']:.2f}")
        price_cols[1].metric("GSA Low", "N/A" if pd.isna(row["GSA Low"]) else f"${row['GSA Low']:.2f}")
        price_cols[2].metric("Suggested Bid", "N/A" if pd.isna(row["Suggested Unit Bid"]) else f"${row['Suggested Unit Bid']:.2f}")
        price_cols[3].metric("Gross Margin", "N/A" if pd.isna(row["Estimated Gross Margin"]) else f"{row['Estimated Gross Margin']:.1%}")

        st.write(f"**Distributor Vendor:** {row['Distributor Vendor'] or 'None matched'}")
        st.write(f"**Price Verdict:** {row['Price Verdict']}")

        st.subheader("Response Readiness")
        st.write(f"**Difficulty:** {row['Difficulty']}")
        st.write(f"**Why:** {row['Why Difficulty']}")
        st.write(f"**Detected Requirements:** {row['Response Requirements'] or 'None detected'}")
        st.write(f"**Contact Emails:** {row['Contact Emails'] or 'None detected'}")

        st.subheader("Ranking Explanation")
        st.write(f"**Positive Signals:** {row['Positive Signals']}")
        st.write(f"**Pass/Risk Reasons:** {row['Pass/Risk Reasons']}")

        st.subheader("Attachments")
        links = row["_attachment_links"]
        if not links:
            st.info("No API attachment links were returned.")
        else:
            for link in links:
                st.markdown(f"- [{link['name']}]({link['url']})")
            st.write(f"**Attachment Status:** {row['Attachment Status'] or 'Not scanned'}")

        with st.expander("Attachment text preview used for scoring"):
            preview = row["_attachment_preview"]
            if preview:
                st.text_area("Attachment Preview", value=preview, height=300)
            else:
                st.info("No readable attachment text was scanned.")
