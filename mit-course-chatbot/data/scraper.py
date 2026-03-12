"""
MIT Course Catalog Scraper
Scrapes https://student.mit.edu/catalog/ and outputs courses.json
"""
from __future__ import annotations

import json
import re
import time
import urllib.request
from collections import defaultdict
from html.parser import HTMLParser

BASE_URL = "https://student.mit.edu/catalog/"
INDEX_URL = BASE_URL + "index.cgi"

DEPARTMENT_PAGES = [
    ("m1a.html",   "Civil and Environmental Engineering"),
    ("m2a.html",   "Mechanical Engineering"),
    ("m3a.html",   "Materials Science and Engineering"),
    ("m4a.html",   "Architecture"),
    ("m5a.html",   "Chemistry"),
    ("m6a.html",   "Electrical Engineering and Computer Science"),
    ("m7a.html",   "Biology"),
    ("m8a.html",   "Physics"),
    ("m9a.html",   "Brain and Cognitive Sciences"),
    ("m10a.html",  "Chemical Engineering"),
    ("m11a.html",  "Urban Studies and Planning"),
    ("m12a.html",  "Earth, Atmospheric, and Planetary Sciences"),
    ("m14a.html",  "Economics"),
    ("m15a.html",  "Management"),
    ("m16a.html",  "Aeronautics and Astronautics"),
    ("m17a.html",  "Political Science"),
    ("m18a.html",  "Mathematics"),
    ("m20a.html",  "Biological Engineering"),
    ("m21a.html",  "Humanities"),
    ("m21Aa.html", "Anthropology"),
    ("mCMSa.html", "Comparative Media Studies"),
    ("m21Wa.html", "Writing"),
    ("m21Ga.html", "Global Languages"),
    ("m21Ha.html", "History"),
    ("m21La.html", "Literature"),
    ("m21Ma.html", "Music"),
    ("m21Ta.html", "Theater Arts"),
    ("mWGSa.html", "Women's and Gender Studies"),
    ("m22a.html",  "Nuclear Science and Engineering"),
    ("m24a.html",  "Linguistics and Philosophy"),
    ("mCCa.html",  "Concourse Program"),
    ("mCGa.html",  "Common Ground for Computing Education"),
    ("mCSBa.html", "Computational and Systems Biology"),
    ("mCSEa.html", "Center for Computational Science and Engineering"),
    ("mECa.html",  "Edgerton Center"),
    ("mEMa.html",  "Engineering and Management"),
    ("mESa.html",  "Experimental Study Group"),
    ("mHSTa.html", "Health Sciences and Technology"),
    ("mIDSa.html", "Institute for Data, Systems and Society"),
    ("mMASa.html", "Media Arts and Sciences"),
    ("mSCMa.html", "Supply Chain Management"),
    ("mASa.html",  "Aerospace Studies"),
    ("mMSa.html",  "Military Science"),
    ("mNSa.html",  "Naval Science"),
    ("mSTSa.html", "Science, Technology, and Society"),
    ("mSWEa.html", "Engineering School-Wide Electives"),
    ("mSPa.html",  "Special Programs"),
]

# Image filenames → semantic meaning
LEVEL_ICONS = {
    "under.gif": "Undergraduate",
    "grad.gif":  "Graduate",
    "ugrad.gif": "Undergraduate",
}
SEMESTER_ICONS = {
    "fall.gif":   "Fall",
    "spring.gif": "Spring",
    "iap.gif":    "IAP",
    "summer.gif": "Summer",
}
DISTRIB_ICONS = {
    "rest.gif":  "REST",
    "hass.gif":  "HASS",
    "hassa.gif": "HASS-A",
    "hassh.gif": "HASS-H",
    "hasss.gif": "HASS-S",
    "hasse.gif": "HASS-E",
    "ci-h.gif":  "CI-H",
    "cih.gif":   "CI-H",
    "ci-hw.gif": "CI-HW",
    "cihw.gif":  "CI-HW",
    "lab.gif":   "Institute Lab",
    "plab.gif":  "Partial Lab",
}

# Schedule bold-label keywords (the catalog uses <b>Label:</b> <i>time</i>)
SCHEDULE_LABELS = re.compile(
    r"^(Lecture|Recitation|Lab|Tutorial|Lec|Rec|Tut)s?:?\s*$", re.I
)


def fetch(url: str, retries: int = 3, delay: float = 1.5) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; MIT-course-scraper/1.0)"}
    req = urllib.request.Request(url, headers=headers)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read()
            try:
                return raw.decode("utf-8")
            except UnicodeDecodeError:
                return raw.decode("latin-1")
        except Exception as exc:
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
            else:
                raise RuntimeError(f"Failed to fetch {url}: {exc}") from exc


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def strip_tags(html: str) -> str:
    """Remove all HTML tags and decode common entities."""
    text = re.sub(r"<[^>]+>", " ", html)
    text = (text
            .replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&nbsp;", " ")
            .replace("&#160;", " ")
            .replace("&quot;", '"')
            .replace("&#39;", "'"))
    return re.sub(r"\s+", " ", text).strip()


def icon_filename(src: str) -> str:
    return src.rsplit("/", 1)[-1].lower()


# ---------------------------------------------------------------------------
# Split a course block (raw HTML) into <br>-delimited logical lines
# ---------------------------------------------------------------------------

def split_br_lines(html: str) -> list[str]:
    """
    The MIT catalog puts all course metadata inside one big <p> block,
    separated by <br> tags.  Split on <br> (any variant) and return
    the resulting HTML fragments.
    """
    parts = re.split(r"<br\s*/?>", html, flags=re.I)
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Pre-processor: split raw department HTML into per-course blocks
# ---------------------------------------------------------------------------

def split_into_course_blocks(html: str) -> list[str]:
    """
    Each course is wrapped in <p><h3>…</h3>…</p> (roughly).
    Split on opening <h3> tags to isolate each course block.
    """
    # Find every <h3> start position
    positions = [m.start() for m in re.finditer(r"<h3\b", html, re.I)]
    if not positions:
        return []
    blocks = []
    for i, start in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(html)
        blocks.append(html[start:end])
    return blocks


# ---------------------------------------------------------------------------
# Parse a single course block
# ---------------------------------------------------------------------------

def parse_course_block(block: str, dept_name: str, page_url: str):
    """
    Parse a single course HTML block into a structured dict.
    Returns None if the block doesn't look like a real course entry.
    """
    course: dict = {
        "course_number": None,
        "title": None,
        "department": dept_name,
        "url": None,
        "level": [],
        "semesters_offered": [],
        "distribution_requirements": [],
        "is_joint": False,
        "same_subject_as": [],
        "meets_with": [],
        "credit_not_for": [],
        "prerequisites": None,
        "units": None,
        "schedule": {},
        "schedule_notes": None,          # "Not offered regularly; consult department" etc.
        "description": None,
        "instructors": [],
        "is_new": False,
        "course_url": None,              # external course website if listed
    }

    # ---- 1. Extract h3: course number + title ----
    h3_m = re.search(r"<h3[^>]*>(.*?)</h3>", block, re.I | re.S)
    if not h3_m:
        return None
    h3_text = strip_tags(h3_m.group(1)).strip()

    m = re.match(r"^([A-Z0-9]+(?:\.[A-Z0-9]+)?(?:\[J\])?)\s+(.*)", h3_text, re.I)
    if m:
        course["course_number"] = m.group(1).strip()
        course["title"] = m.group(2).strip()
        course["is_joint"] = "[J]" in course["course_number"]
    else:
        # No recognisable number; skip
        return None

    # Reject section headers that look like plain words (e.g. "Undergraduate", "Introductory")
    # Valid course numbers contain at least one digit
    if not re.search(r"\d", course["course_number"]):
        return None

    anchor = course["course_number"].replace("[J]", "").strip()
    course["url"] = page_url + "#" + anchor

    # ---- 2. Icons (level / semester / distribution) ----
    for img_m in re.finditer(r'<img\b[^>]+>', block, re.I):
        img_tag = img_m.group(0)
        src_m = re.search(r'src="([^"]+)"', img_tag, re.I)
        if not src_m:
            continue
        fname = icon_filename(src_m.group(1))
        title_m = re.search(r'title="([^"]*)"', img_tag, re.I)
        if fname in LEVEL_ICONS:
            lv = LEVEL_ICONS[fname]
            if lv not in course["level"]:
                course["level"].append(lv)
        elif fname in SEMESTER_ICONS:
            sem = SEMESTER_ICONS[fname]
            if sem not in course["semesters_offered"]:
                course["semesters_offered"].append(sem)
        elif fname in DISTRIB_ICONS:
            dr = DISTRIB_ICONS[fname]
            if dr not in course["distribution_requirements"]:
                course["distribution_requirements"].append(dr)

    # ---- 3. Strip the h3 and work with the body ----
    body = block[h3_m.end():]

    # ---- 4. Parse br-delimited lines ----
    lines = split_br_lines(body)
    desc_lines: list[str] = []

    for line_html in lines:
        text = strip_tags(line_html).strip()
        if not text:
            continue

        # Skip icon-only lines (just images)
        if re.match(r"^[\s(),.]+$", text):
            continue

        # ---- (New) ----
        if re.search(r"\(New\)", text, re.I):
            course["is_new"] = True
            continue

        # ---- Not offered ----
        if re.search(r"not offered regularly|consult department", text, re.I):
            course["schedule_notes"] = text
            continue

        # ---- Same subject as ----
        if re.search(r"same subject as", text, re.I):
            # Prefer link text (visible course numbers) over href anchors
            refs = re.findall(r"<a\b[^>]*>([^<]+)</a>", line_html)
            refs = [r.strip() for r in refs if re.search(r"\d", r)]
            if not refs:
                refs = re.findall(r'href="[^"]*#([^"]+)"', line_html)
            course["same_subject_as"] = refs or []
            continue

        # ---- Subject meets with ----
        if re.search(r"subject meets with", text, re.I):
            refs = re.findall(r"<a\b[^>]*>([^<]+)</a>", line_html)
            refs = [r.strip() for r in refs if re.search(r"\d", r)]
            if not refs:
                refs = re.findall(r'href="[^"]*#([^"]+)"', line_html)
            course["meets_with"] = refs or []
            continue

        # ---- Credit cannot also be received for ----
        if re.search(r"credit cannot", text, re.I):
            refs = re.findall(r'href="[^"]*#([^"]+)"', line_html)
            course["credit_not_for"] = refs or [text]
            continue

        # ---- Prerequisites  (bold label: <b>Prereq:</b> or plain text) ----
        if re.search(r"<b>Prereq", line_html, re.I) or re.match(r"Prereq", text, re.I):
            # Remove the label part
            val = re.sub(r"^<b>Prereq(?:uisite)?s?:?</b>\s*", "", line_html, flags=re.I)
            val = re.sub(r"^Prereq(?:uisite)?s?:?\s*", "", strip_tags(val), flags=re.I).strip()
            course["prerequisites"] = val if val else None
            continue

        # ---- Units ----
        if re.search(r"<b>Units?:?</b>", line_html, re.I) or re.match(r"Units?:", text, re.I):
            val = re.sub(r"^<b>Units?:?</b>\s*", "", line_html, flags=re.I)
            val = re.sub(r"^Units?:?\s*", "", strip_tags(val), flags=re.I).strip()
            course["units"] = val if val else None
            continue

        # ---- External URL ----
        if re.match(r"URL:", text, re.I):
            url_m = re.search(r'href="([^"]+)"', line_html)
            if url_m:
                course["course_url"] = url_m.group(1)
            continue

        # ---- Schedule line: contains <b>Lecture:</b>, <b>Recitation:</b>, <b>Lab:</b> etc. ----
        sched_labels = re.findall(r"<b>(Lecture|Recitation|Lab|Tutorial|Lec|Rec|Tut)s?:?</b>", line_html, re.I)
        if sched_labels:
            # Split on each <b>Label:</b> occurrence
            parts = re.split(r"<b>(?:Lecture|Recitation|Lab|Tutorial|Lec|Rec|Tut)s?:?</b>", line_html, flags=re.I)
            labels = re.findall(r"<b>(Lecture|Recitation|Lab|Tutorial|Lec|Rec|Tut)s?:?</b>", line_html, re.I)
            for label, part in zip(labels, parts[1:]):
                label_clean = label.strip(":").capitalize()
                # Normalise label synonyms
                if label_clean.lower() in ("lec",):
                    label_clean = "Lecture"
                elif label_clean.lower() in ("rec",):
                    label_clean = "Recitation"
                elif label_clean.lower() in ("tut",):
                    label_clean = "Tutorial"
                time_val = strip_tags(part).strip()
                # Clean up "+final" and trailing noise
                time_val = re.sub(r"\s*\+final\b", " +final", time_val, flags=re.I).strip()
                if time_val:
                    course["schedule"][label_clean] = time_val
            continue

        # ---- Instructor: italic <i>Name</i>, possibly preceded by Fall:/Spring: ----
        i_tags = re.findall(r"<i>(.*?)</i>", line_html, re.S)
        if i_tags:
            found_instructor = False
            for it in i_tags:
                name = strip_tags(it).strip()
                # Skip time strings (start with day letters + digit, or contain digits with dots)
                if not name:
                    continue
                if re.match(r"^[MTWRFS]+\d", name):
                    continue
                if re.search(r"\+final", name, re.I):
                    continue
                if re.match(r"^\d", name):
                    continue
                if name.lower() in ("fall", "spring", "iap", "summer", "tba"):
                    continue
                # Looks like a name
                if name not in course["instructors"]:
                    course["instructors"].append(name)
                    found_instructor = True
            if found_instructor:
                continue

        # ---- Description: catch-all plain text lines ----
        # Skip very short / noise lines
        if len(text) < 20:
            continue
        if text.startswith("No textbook"):
            continue
        # Skip lines that are just href anchor dumps
        if re.match(r"^https?://", text):
            continue
        desc_lines.append(text)

    # Assemble description from collected plain-text lines
    if desc_lines:
        course["description"] = " ".join(desc_lines)

    # ---- 5. Determine schedule_notes fallback ----
    # If no schedule was found and no explicit "not offered" note, mark appropriately
    if not course["schedule"] and course["schedule_notes"] is None:
        course["schedule_notes"] = "Schedule not listed — contact department or check MIT course catalog directly"

    return course


# ---------------------------------------------------------------------------
# Main scraping logic
# ---------------------------------------------------------------------------

def scrape_department(page: str, dept_name: str) -> list[dict]:
    url  = BASE_URL + page
    html = fetch(url)
    blocks = split_into_course_blocks(html)
    courses = []
    for block in blocks:
        c = parse_course_block(block, dept_name, url)
        if c:
            courses.append(c)
    return courses


def merge_duplicates(courses: list[dict]) -> list[dict]:
    """
    Merge courses that share the same course_number across departments.

    Rules:
    - If two entries have the same number AND the same title (cross-listings):
      merge into one entry, combining department names into a list under
      'departments' and picking the richer copy for each field.
    - If two entries have the same number but different titles (genuinely
      different courses, e.g. UPOP): keep both, append a suffix _A/_B.
    - Exact same-dept duplicates (e.g. 21M.S55 appearing twice): keep one.
    """
    from collections import OrderedDict

    # Group by course_number
    groups: dict[str, list[dict]] = OrderedDict()
    for c in courses:
        groups.setdefault(c["course_number"], []).append(c)

    merged: list[dict] = []
    for num, group in groups.items():
        if len(group) == 1:
            # Ensure 'departments' list field exists for consistency
            c = group[0]
            c["departments"] = [c.pop("department")]
            merged.append(c)
            continue

        # Deduplicate exact same-dept copies first
        seen_depts: set[str] = set()
        unique: list[dict] = []
        for c in group:
            key = (c["department"], c["title"])
            if key not in seen_depts:
                seen_depts.add(key)
                unique.append(c)
        group = unique

        if len(group) == 1:
            c = group[0]
            c["departments"] = [c.pop("department")]
            merged.append(c)
            continue

        # Check if all entries share the same title (cross-listing)
        titles = {c["title"] for c in group}
        if len(titles) == 1:
            # True cross-listing — merge into one, keeping richest field values
            base = group[0].copy()
            base["departments"] = [c["department"] for c in group]
            base.pop("department", None)
            for other in group[1:]:
                # Prefer non-None / non-empty values from either copy
                for field in ("prerequisites", "units", "description", "schedule_notes", "course_url"):
                    if not base.get(field) and other.get(field):
                        base[field] = other[field]
                for field in ("schedule",):
                    if not base.get(field) and other.get(field):
                        base[field] = other[field]
                for field in ("level", "semesters_offered", "distribution_requirements",
                              "same_subject_as", "meets_with", "credit_not_for", "instructors"):
                    combined = base.get(field, []) + [
                        x for x in other.get(field, []) if x not in base.get(field, [])
                    ]
                    base[field] = combined
                if other.get("is_new"):
                    base["is_new"] = True
            merged.append(base)
        else:
            # Different titles under same number — keep both, tag with suffix
            for i, c in enumerate(group):
                c = c.copy()
                suffix = chr(ord("A") + i)
                c["course_number"] = f"{num}_{suffix}"
                c["departments"] = [c.pop("department")]
                merged.append(c)

    return merged


def scrape_all() -> list[dict]:
    all_courses = []
    for page, dept_name in DEPARTMENT_PAGES:
        print(f"  Scraping {dept_name} ({page})...", flush=True)
        try:
            courses = scrape_department(page, dept_name)
            print(f"    → {len(courses)} courses", flush=True)
            all_courses.extend(courses)
        except Exception as exc:
            print(f"    ERROR: {exc}", flush=True)
        time.sleep(0.5)
    all_courses = merge_duplicates(all_courses)
    print(f"\nAfter merging duplicates: {len(all_courses)} unique courses")
    return all_courses


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def print_stats(courses: list[dict]):
    print("\n" + "=" * 60)
    print("SCRAPE SUMMARY")
    print("=" * 60)
    print(f"Total courses scraped : {len(courses)}")

    dept_counts: dict[str, int] = defaultdict(int)
    for c in courses:
        for dept in c.get("departments", [c.get("department", "Unknown")]):
            dept_counts[dept] += 1
    print("\nCourses per department (cross-listed counted in each):")
    for dept, count in sorted(dept_counts.items(), key=lambda x: -x[1]):
        print(f"  {count:4d}  {dept}")

    missing_desc   = sum(1 for c in courses if not c.get("description"))
    missing_prereq = sum(1 for c in courses if c.get("prerequisites") is None)
    has_schedule   = sum(1 for c in courses if c.get("schedule"))
    not_offered    = sum(1 for c in courses if (c.get("schedule_notes") or "").startswith("Not offered"))
    no_sched_info  = sum(1 for c in courses if not c.get("schedule") and
                         not (c.get("schedule_notes") or "").startswith("Not offered"))
    missing_units  = sum(1 for c in courses if not c.get("units"))
    missing_instr  = sum(1 for c in courses if not c.get("instructors"))
    joint_courses  = sum(1 for c in courses if c.get("is_joint"))
    new_courses    = sum(1 for c in courses if c.get("is_new"))
    total          = len(courses)

    print(f"\nMissing descriptions  : {missing_desc}  ({100*missing_desc/max(total,1):.1f}%)")
    print(f"Missing prereqs       : {missing_prereq}  ({100*missing_prereq/max(total,1):.1f}%)")
    print(f"Have explicit schedule: {has_schedule}  ({100*has_schedule/max(total,1):.1f}%)")
    print(f"'Not offered' note    : {not_offered}")
    print(f"No schedule info      : {no_sched_info}  ({100*no_sched_info/max(total,1):.1f}%)")
    print(f"Missing units         : {missing_units}  ({100*missing_units/max(total,1):.1f}%)")
    print(f"Missing instructors   : {missing_instr}  ({100*missing_instr/max(total,1):.1f}%)")
    print(f"\nJoint ([J]) courses   : {joint_courses}")
    print(f"New courses           : {new_courses}")

    distrib_counts: dict[str, int] = defaultdict(int)
    for c in courses:
        for dr in c.get("distribution_requirements", []):
            distrib_counts[dr] += 1
    if distrib_counts:
        print("\nDistribution requirements:")
        for dr, cnt in sorted(distrib_counts.items()):
            print(f"  {dr:<12} {cnt}")

    sem_counts: dict[str, int] = defaultdict(int)
    for c in courses:
        for s in c.get("semesters_offered", []):
            sem_counts[s] += 1
    if sem_counts:
        print("\nSemesters offered:")
        for s, cnt in sorted(sem_counts.items()):
            print(f"  {s:<12} {cnt}")

    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    out_dir  = os.path.dirname(os.path.abspath(__file__))
    out_file = os.path.join(out_dir, "courses.json")

    print("MIT Course Catalog Scraper")
    print(f"Output: {out_file}\n")

    courses = scrape_all()
    print_stats(courses)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(courses, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {len(courses)} courses to {out_file}")
