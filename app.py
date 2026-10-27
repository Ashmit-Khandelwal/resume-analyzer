import re
import pdfplumber
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Resume Analyzer", page_icon="📄", layout="wide")

HOST_URL = "https://your-website-link.com"   # replace with your real link
HOST_NAME = "Visit Host Page"

# ---------- STYLE ----------
st.markdown("""
<style>
.hero {
    background: linear-gradient(120deg, #4f46e5, #7c3aed, #db2777);
    padding: 2.2rem 2rem; border-radius: 18px; color: white; margin-bottom: 1.5rem;
}
.hero h1 { margin: 0; font-size: 2.3rem; color: white; }
.hero p { margin: .4rem 0 0; opacity: .92; font-size: 1.05rem; }
.card {
    background: rgba(127,127,127,.10); border: 1px solid rgba(127,127,127,.25);
    border-radius: 14px; padding: 1.1rem 1.2rem; text-align: center; height: 100%;
}
.card .num { font-size: 2rem; font-weight: 700; }
.card .lbl { opacity: .75; font-size: .9rem; }
.gauge {
    width: 170px; height: 170px; border-radius: 50%; margin: auto;
    display: flex; align-items: center; justify-content: center;
}
.gauge-inner {
    width: 128px; height: 128px; border-radius: 50%; background: var(--background-color, #0e1117);
    display: flex; flex-direction: column; align-items: center; justify-content: center;
}
.gauge-inner b { font-size: 2.1rem; }
.chip {
    display: inline-block; padding: 5px 13px; margin: 4px 4px 4px 0;
    border-radius: 999px; font-size: .88rem; font-weight: 600;
}
.good { background: rgba(34,197,94,.18); color: #16a34a; border: 1px solid #22c55e; }
.bad  { background: rgba(239,68,68,.15); color: #dc2626; border: 1px solid #ef4444; }
.info { background: rgba(99,102,241,.15); color: #6366f1; border: 1px solid #6366f1; }
.stButton > button {
    background: linear-gradient(120deg, #4f46e5, #db2777); color: white;
    border: none; border-radius: 10px; padding: .6rem 1.6rem; font-weight: 600;
}

.hero a.hero-btn {
    display: inline-block; margin-top: 1rem; padding: .55rem 1.3rem;
    background: white; color: #4f46e5 !important; font-weight: 700;
    border-radius: 999px; text-decoration: none;
}
.hero a.hero-btn:hover { opacity: .9; }
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="hero">
  <h1>📄 AI Resume Analyzer</h1>
  <p>Upload your resume, paste a job description, and see exactly what is missing.</p>
  <a class="hero-btn" href="{HOST_URL}" target="_blank">🔗 {HOST_NAME}</a>
</div>
""", unsafe_allow_html=True)

# ---------- DATA ----------
SKILLS = [
    "python", "java", "javascript", "typescript", "c++", "c#", "sql", "mysql",
    "postgresql", "mongodb", "html", "css", "react", "angular", "vue", "node.js",
    "django", "flask", "fastapi", "spring", "git", "github", "docker", "kubernetes",
    "aws", "azure", "gcp", "linux", "rest api", "api", "machine learning",
    "deep learning", "nlp", "pandas", "numpy", "scikit-learn", "tensorflow",
    "pytorch", "data analysis", "data structures", "algorithms", "excel",
    "power bi", "tableau", "ci/cd", "agile", "scrum", "flutter", "android",
    "kotlin", "firebase", "communication", "teamwork", "leadership", "problem solving",
]
SECTIONS = {
    "Education": ["education", "university", "college", "b.tech", "bachelor"],
    "Experience / Internships": ["experience", "internship", "work history"],
    "Projects": ["projects", "project"],
    "Skills": ["skills", "technologies", "tools"],
    "Certifications / Achievements": ["certification", "certificate", "achievement", "award"],
}
FILLER = {
    "ability", "strong", "team", "work", "working", "years", "year", "experience",
    "required", "preferred", "looking", "role", "job", "candidate", "company",
    "responsibilities", "requirements", "skills", "knowledge", "good", "excellent",
    "including", "etc", "will", "using", "use", "new", "build", "building",
    "develop", "development", "understanding", "familiarity", "plus", "must",
}
ACTION_VERBS = [
    "developed", "built", "designed", "implemented", "created", "led", "managed",
    "improved", "optimized", "analyzed", "automated", "deployed", "launched",
    "reduced", "increased", "achieved", "collaborated",
]


# ---------- HELPERS ----------
def extract_text(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += (page.extract_text() or "") + "\n"
    return text


def clean(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9+#\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def has_term(text, term):
    pattern = r"(?<![a-z0-9+#])" + re.escape(term) + r"(?![a-z0-9+#])"
    return re.search(pattern, text) is not None


def match_score(resume, job_desc):
    vec = TfidfVectorizer(stop_words="english")
    m = vec.fit_transform([resume, job_desc])
    return cosine_similarity(m[0], m[1])[0][0] * 100


def other_keywords(resume, job_desc, top_n=15):
    vec = TfidfVectorizer(stop_words="english")
    tfidf = vec.fit_transform([job_desc])
    ranked = sorted(zip(vec.get_feature_names_out(), tfidf.toarray()[0]),
                    key=lambda x: x[1], reverse=True)
    words = set(resume.split())
    return [w for w, _ in ranked
            if w not in words and w not in FILLER and len(w) > 2 and not w.isdigit()][:top_n]


def chips(items, css):
    return "".join(f'<span class="chip {css}">{i}</span>' for i in items)


def card(num, label):
    return f'<div class="card"><div class="num">{num}</div><div class="lbl">{label}</div></div>'


def analyze(raw_resume, job_desc):
    r_low, j_low = raw_resume.lower(), job_desc.lower()
    r_clean, j_clean = clean(raw_resume), clean(job_desc)

    jd_skills = [s for s in SKILLS if has_term(j_low, s)]
    have = [s for s in jd_skills if has_term(r_low, s)]
    missing = [s for s in jd_skills if not has_term(r_low, s)]

    sections_found = [n for n, k in SECTIONS.items() if any(has_term(r_low, x) for x in k)]
    sections_missing = [n for n in SECTIONS if n not in sections_found]

    contact = {
        "Email": bool(re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", raw_resume)),
        "Phone": bool(re.search(r"\+?\d[\d\s\-]{8,}\d", raw_resume)),
        "LinkedIn": "linkedin" in r_low,
        "GitHub": "github" in r_low,
    }

    words = len(raw_resume.split())
    verbs_used = [v for v in ACTION_VERBS if has_term(r_low, v)]
    numbers = len(re.findall(r"\d+\s?%|\d+\+|\b\d{2,}\b", raw_resume))

    tips = []
    if words < 250:
        tips.append("Your resume looks short. Add more detail about projects and experience.")
    elif words > 900:
        tips.append("Your resume is long. Try to keep it to 1 page (2 at most).")
    if len(verbs_used) < 4:
        tips.append("Start bullet points with action verbs such as: developed, built, designed, optimized.")
    if numbers < 3:
        tips.append("Add numbers to show impact (for example: 'improved speed by 30%').")
    if missing:
        tips.append("Add the missing skills you genuinely have to your Skills section.")
    if sections_missing:
        tips.append("Add the missing sections: " + ", ".join(sections_missing) + ".")
    if not all(contact.values()):
        tips.append("Complete your contact details (email, phone, LinkedIn, GitHub).")
    if not tips:
        tips.append("Great resume! Only small tweaks are needed.")

    return {
        "score": match_score(r_clean, j_clean),
        "have": have, "missing": missing, "jd_skills": jd_skills,
        "sections_found": sections_found, "sections_missing": sections_missing,
        "contact": contact, "words": words, "verbs": verbs_used, "numbers": numbers,
        "others": [w for w in other_keywords(r_clean, j_clean) if w not in missing],
        "tips": tips,
    }


def build_report(r):
    lines = ["RESUME ANALYSIS REPORT", "=" * 30,
             f"Match score: {r['score']:.1f}%", "",
             "Skills you have: " + (", ".join(r["have"]) or "none"),
             "Missing skills: " + (", ".join(r["missing"]) or "none"),
             "Missing sections: " + (", ".join(r["sections_missing"]) or "none"),
             "Missing contact details: " + (", ".join(k for k, v in r["contact"].items() if not v) or "none"),
             "Other missing keywords: " + (", ".join(r["others"]) or "none"), "",
             "Suggestions:"]
    lines += [f"- {t}" for t in r["tips"]]
    return "\n".join(lines)


# ---------- SIDEBAR ----------
with st.sidebar:
    st.header("ℹ️ How to use")
    st.markdown("1. Upload your resume (PDF)\n2. Paste the job description\n3. Click **Analyze**")
    st.divider()
    st.header("💡 Quick tips")
    st.markdown("- Keep it to 1 page\n- Use action verbs\n- Add numbers and results\n"
                "- Tailor it for every job")
    st.divider()
    st.caption("Tip: scores of 40-60% are normal for a good fit.")

    st.divider()
    st.link_button(f"🔗 {HOST_NAME}", HOST_URL)

# ---------- INPUT ----------
c1, c2 = st.columns(2)
with c1:
    st.subheader("1️⃣ Your resume")
    uploaded = st.file_uploader("Upload resume (PDF)", type=["pdf"])
with c2:
    st.subheader("2️⃣ Job description")
    job_desc = st.text_area("Paste the job description", height=160,
                            label_visibility="collapsed",
                            placeholder="Paste the full job description here...")

if st.button("🔍 Analyze my resume"):
    if not uploaded or not job_desc.strip():
        st.warning("Please upload a resume and paste a job description.")
    else:
        with st.spinner("Analyzing your resume..."):
            raw = extract_text(uploaded)
            if not clean(raw):
                st.error("Couldn't read text from this PDF. It may be a scanned image.")
                st.session_state.pop("result", None)
            else:
                st.session_state["result"] = analyze(raw, job_desc)

# ---------- RESULTS ----------
r = st.session_state.get("result")
if r:
    st.divider()
    score = r["score"]
    color = "#ef4444" if score < 30 else "#f59e0b" if score < 50 else "#22c55e"
    verdict = ("Low match: needs a lot of changes." if score < 30 else
               "Average match: add the missing items to improve." if score < 50 else
               "Good match: a few small additions will make it stronger.")

    g, m1, m2, m3 = st.columns([1.2, 1, 1, 1])
    with g:
        st.markdown(
            f'<div class="gauge" style="background: conic-gradient({color} {score}%, rgba(127,127,127,.25) 0);">'
            f'<div class="gauge-inner"><b>{score:.0f}%</b><span>match</span></div></div>',
            unsafe_allow_html=True)
        st.markdown(f"<p style='text-align:center;margin-top:.6rem'>{verdict}</p>",
                    unsafe_allow_html=True)
    skill_pct = f"{len(r['have'])}/{len(r['jd_skills'])}" if r["jd_skills"] else "N/A"
    m1.markdown(card(skill_pct, "Required skills you have"), unsafe_allow_html=True)
    m2.markdown(card(f"{len(r['sections_found'])}/{len(SECTIONS)}", "Resume sections found"),
                unsafe_allow_html=True)
    m3.markdown(card(f"{sum(r['contact'].values())}/4", "Contact details present"),
                unsafe_allow_html=True)

    st.write("")
    t1, t2, t3, t4, t5 = st.tabs(
        ["🛠 Skills", "📑 Sections & Contact", "🔑 Keywords", "💡 Suggestions", "⬇️ Report"])

    with t1:
        st.markdown("#### ❌ Missing skills")
        st.markdown(chips(r["missing"], "bad") if r["missing"]
                    else "✅ You have every skill the job mentions.", unsafe_allow_html=True)
        st.markdown("#### ✅ Skills you already have")
        st.markdown(chips(r["have"], "good") if r["have"]
                    else "No matching skills found.", unsafe_allow_html=True)
        if r["jd_skills"]:
            st.markdown("#### Skill coverage")
            st.progress(len(r["have"]) / len(r["jd_skills"]))

    with t2:
        a, b = st.columns(2)
        with a:
            st.markdown("#### Sections")
            st.markdown(chips(r["sections_found"], "good") + chips(r["sections_missing"], "bad"),
                        unsafe_allow_html=True)
            st.caption("Green = found, red = missing")
        with b:
            st.markdown("#### Contact details")
            st.markdown("".join(
                f'<span class="chip {"good" if ok else "bad"}">{"✔" if ok else "✘"} {k}</span>'
                for k, ok in r["contact"].items()), unsafe_allow_html=True)

    with t3:
        st.markdown("#### Important words from the job description missing in your resume")
        st.markdown(chips(r["others"], "info") if r["others"]
                    else "✅ No other important keywords are missing.", unsafe_allow_html=True)
        st.caption("Add only the words that truly describe your work.")

    with t4:
        for t in r["tips"]:
            st.info(t)
        x, y, z = st.columns(3)
        x.metric("Word count", r["words"])
        y.metric("Action verbs used", len(r["verbs"]))
        z.metric("Numbers / metrics", r["numbers"])

    with t5:
        st.write("Download a summary of this analysis.")
        st.download_button("⬇️ Download report (.txt)", build_report(r),
                           file_name="resume_report.txt")