import streamlit as st
import json
import pandas as pd
import os
import re
from datetime import datetime

# Import scoring functions
from scoring import (
    score_candidate, 
    parse_job_description, 
    get_embedding_model, 
    CURRENT_DATE,
    parse_date,
    sk_name_proper
)

# Set page config for a premium dark mode recruiting app
st.set_page_config(
    page_title="Redrob Talent Intelligence",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Lazy load Plotly
try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# Custom premium CSS styling (glassmorphism headers, modern cards, harmonious color scheme)
st.markdown("""
<style>
    /* Dark theme overrides */
    .stApp {
        background-color: #0b0f19;
        color: #f3f4f6;
    }
    
    /* Title and main headers */
    h1, h2, h3, h4 {
        font-family: 'Outfit', 'Inter', sans-serif !important;
        color: #ffffff;
    }
    
    .main-title-container {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        color: #9ca3af;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Cards and containers */
    .metric-card {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: transform 0.2s, border-color 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #3b82f6;
    }
    
    .stat-number {
        font-size: 2.2rem;
        font-weight: 700;
        color: #3b82f6;
        margin-bottom: 0.2rem;
    }
    
    .stat-label {
        font-size: 0.85rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Verified badges */
    .badge-verified {
        background-color: #065f46;
        color: #34d399;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    .badge-unverified {
        background-color: #7f1d1d;
        color: #f87171;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    /* Career history timeline cards */
    .timeline-card {
        background-color: #1e293b;
        border-left: 4px solid #8b5cf6;
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 0 8px 8px 0;
    }
    
    .timeline-title {
        font-weight: 700;
        font-size: 1.05rem;
        color: #ffffff;
    }
    
    .timeline-meta {
        font-size: 0.85rem;
        color: #9ca3af;
        margin-bottom: 0.5rem;
    }
    
    /* Copilot boxes */
    .copilot-box {
        background-color: #111827;
        border: 1px solid #3b82f6;
        border-radius: 8px;
        padding: 1rem;
        font-family: monospace;
        color: #38bdf8;
        white-space: pre-wrap;
    }
</style>
""", unsafe_allow_html=True)

# Main Title and Header
st.markdown("<div class='main-title-container'>Redrob AI Talent Intelligence</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Vetting & ranking candidates by real-world capability, behavioral signals, and profile integrity.</div>", unsafe_allow_html=True)

# Sidebar - Configuration and Data Upload
st.sidebar.image("https://static.wixstatic.com/media/2cd43b_55b3eb2b89f84852b7b51d38eb15e5a2~mv2.png/v1/fill/w_320,h_100,al_c,q_85,usm_0.66_1.00_0.01,enc_auto/2cd43b_55b3eb2b89f84852b7b51d38eb15e5a2~mv2.png", width=180)

st.sidebar.markdown("### 1. Job Description")
default_jd = (
    "We are looking for a Senior AI/ML Engineer with 5 to 9 years of experience. "
    "The ideal candidate should have strong expertise in embeddings, dense retrieval, and vector search. "
    "Hands-on experience with vector databases like Pinecone, Milvus, or Qdrant is required. "
    "You should be well-versed in LLM fine-tuning using PEFT, LoRA, or QLoRA, and "
    "familiar with offline ranking evaluation metrics like NDCG, MRR, and MAP."
)
jd_text = st.sidebar.text_area("Paste the target Job Description:", value=default_jd, height=150)
jd_info = parse_job_description(jd_text)

# Display extracted info in sidebar
with st.sidebar.expander("🔍 Extracted JD Criteria", expanded=False):
    st.write(f"**Target YoE**: {jd_info['min_yoe']} - {jd_info['max_yoe']} years")
    st.write(f"**Target Titles**: {', '.join(jd_info['titles'])}")
    st.write(f"**Key Skills**: {', '.join([sk_name_proper(s) for s in jd_info['skills']])}")

st.sidebar.markdown("### 2. Vetting Weights")
w_skills = st.sidebar.slider("Technical Skills & Similarity (%)", 0, 100, 35)
w_exp = st.sidebar.slider("Experience & Title Alignment (%)", 0, 100, 25)
w_loc = st.sidebar.slider("Location Preference (%)", 0, 100, 15)
w_beh = st.sidebar.slider("Behavior & Availability (%)", 0, 100, 25)

# Normalize weights
total_weight = w_skills + w_exp + w_loc + w_beh
if total_weight == 0:
    weights = {"skills": 0.25, "experience": 0.25, "location": 0.25, "behavior": 0.25}
else:
    weights = {
        "skills": w_skills / total_weight,
        "experience": w_exp / total_weight,
        "location": w_loc / total_weight,
        "behavior": w_beh / total_weight
    }

if total_weight != 100 and total_weight > 0:
    st.sidebar.caption(f"⚠️ Weights normalized to sum to 100% (currently sums to {total_weight}%)")

st.sidebar.markdown("### 3. Model Configuration")
model_mode = st.sidebar.radio(
    "Semantic Scoring Model:",
    ["Fast Lexical (TF-IDF / Heuristics)", "Deep Semantic (Sentence-Transformers)"]
)

# File Uploader
st.sidebar.markdown("### 4. Upload Candidate Pool")
uploaded_file = st.sidebar.file_uploader("Upload candidates file (JSONL, JSON, or GZ)", type=["jsonl", "json", "gz"])

default_candidates_path = "./[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/sample_candidates.json"

@st.cache_data
def load_candidates_data(file_obj, path=None):
    import gzip
    candidates = []
    if file_obj is not None:
        if file_obj.name.endswith(".gz"):
            with gzip.open(file_obj, "rt", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        candidates.append(json.loads(line))
        else:
            content = file_obj.read().decode("utf-8")
            if file_obj.name.endswith(".jsonl"):
                for line in content.splitlines():
                    if line.strip():
                        candidates.append(json.loads(line))
            else:
                candidates = json.loads(content)
    elif path and os.path.exists(path):
        is_gz = path.endswith(".gz")
        open_f = gzip.open if is_gz else open
        mode = "rt" if is_gz else "r"
        with open_f(path, mode, encoding="utf-8") as f:
            if path.endswith(".jsonl") or path.endswith(".jsonl.gz"):
                for line in f:
                    if line.strip():
                        candidates.append(json.loads(line))
            else:
                candidates = json.load(f)
    return candidates

# Helper generators for AI Recruiter Copilot
def generate_outreach_email(profile, jd_info, score):
    name = profile.get("anonymized_name", "Candidate")
    title = profile.get("current_title", "Software Engineer")
    company = profile.get("current_company", "your current company")
    skills_matched = [sk_name_proper(s) for s in jd_info.get("skills", [])[:3]]
    
    email_template = f"""Subject: Exciting Opportunity: Senior AI/ML Role | Match Score: {round(score, 1)}%

Dear {name},

I hope this email finds you well. 

I came across your profile and was highly impressed by your background as a {title}{f" at {company}" if company else ""}. Your experience aligns closely with our hiring requirements.

We are currently building out our core AI team and looking for experts with hands-on experience in {', '.join(skills_matched[:-1])} and {skills_matched[-1]}. Given your excellent background, I believe you would be a fantastic fit for this position.

I would love to connect for a brief 15-minute introductory call to share more about our vision and learn about your career goals. Are you available sometime this week?

Best regards,
Talent Acquisition Team
Redrob Intelligence
"""
    return email_template

def generate_interview_questions(profile, jd_info):
    skills = [s.lower() for s in jd_info.get("skills", [])]
    questions = []
    
    if "embeddings" in skills or "sentence-transformers" in skills or "sentence transformers" in skills:
        questions.append("1. Can you explain how you would select and fine-tune an embedding model (like BGE or E5) for a custom domain dataset? What evaluation metrics would you track?")
    if "vector search" in skills or "vector db" in skills:
        questions.append("2. In a production environment with millions of vectors, how do you handle vector database scaling and index optimization (e.g., HNSW vs IVF-PQ)?")
    if "evaluation" in skills or "ndcg" in skills or "mrr" in skills:
        questions.append("3. How do you design an offline-online evaluation loop for a ranking model? How do you ensure that improvements in offline NDCG translate to better online A/B testing metrics?")
    if "fine-tuning" in skills or "lora" in skills or "qlora" in skills:
        questions.append("4. What are the key differences in resource requirements and model performance when fine-tuning a model using LoRA versus full parameter fine-tuning?")
        
    while len(questions) < 3:
        q_num = len(questions) + 1
        questions.append(f"{q_num}. Can you describe a challenging technical problem you solved in your recent role and how you measured its impact?")
        
    return "\n\n".join(questions[:3])

# Load data
candidates = []
data_source = ""
if uploaded_file is not None:
    candidates = load_candidates_data(uploaded_file)
    data_source = f"Uploaded File ({len(candidates)} candidates)"
elif os.path.exists(default_candidates_path):
    candidates = load_candidates_data(None, default_candidates_path)
    data_source = f"Loaded Sample Pool ({len(candidates)} candidates)"

if not candidates:
    st.info("Please upload a candidates file in the sidebar to begin.")
else:
    # Set up models if needed
    model = None
    jd_embedding = None
    if model_mode == "Deep Semantic (Sentence-Transformers)":
        with st.spinner("Loading local Sentence-Transformer model (all-MiniLM-L6-v2)..."):
            model = get_embedding_model()
            if model:
                jd_embedding = model.encode(jd_text, convert_to_tensor=True)
            else:
                st.sidebar.error("Failed to load Sentence-Transformers. Falling back to Lexical mode.")

    # Two-Stage Scoring Pipeline
    scored_candidates = []
    skipped_anom = 0
    skipped_yoe = 0
    skipped_title = 0
    skipped_consult = 0
    skipped_cv = 0
    
    # Words in JD for fast Stage-1 screening
    stopwords = {"and", "the", "for", "with", "you", "will", "our", "are", "that", "this", "from", "have", "role", "team", "work", "experience", "required", "preferred"}
    jd_words = set(re.findall(r'\b[a-z]{3,15}\b', jd_text.lower())) - stopwords
    
    # Stage 1: Fast Filter & Lexical Overlap
    stage1_pool = []
    for cand in candidates:
        # Fast Vetting Checks (Honeypot detection)
        signals = cand.get("redrob_signals", {})
        signup_dt = parse_date(signals.get("signup_date"))
        last_active_dt = parse_date(signals.get("last_active_date"))
        if signup_dt and last_active_dt and last_active_dt < signup_dt:
            skipped_anom += 1
            continue
            
        sal = signals.get("expected_salary_range_inr_lpa", {})
        if sal.get("min", 0) > sal.get("max", 0):
            skipped_anom += 1
            continue
            
        profile = cand.get("profile", {})
        career = cand.get("career_history", [])
        
        # Simple word overlap
        text = f"{profile.get('current_title', '')} {profile.get('summary', '')} "
        text += " ".join([job.get("description", "") + " " + job.get("title", "") for job in career])
        text = text.lower()
        lex_score = sum(1 for w in jd_words if w in text)
        
        stage1_pool.append((lex_score, cand))
        
    # Sort and take top 500
    stage1_pool.sort(key=lambda x: -x[0])
    top_fits = [item[1] for item in stage1_pool[:500]]
    
    # Stage 2: Deep Re-ranking
    for cand in top_fits:
        score, reason = score_candidate(
            cand, 
            jd_text=jd_text, 
            weights=weights, 
            jd_info=jd_info, 
            model=model, 
            jd_embedding=jd_embedding
        )
        if score > 0:
            scored_candidates.append({
                "cand": cand,
                "score": score,
                "reasoning": reason
            })
        else:
            if "Anomaly:" in reason:
                skipped_anom += 1
            elif "YoE" in reason:
                skipped_yoe += 1
            elif "title" in reason:
                skipped_title += 1
            elif "consulting" in reason:
                skipped_consult += 1
            elif "CV" in reason:
                skipped_cv += 1
                
    # Sort
    scored_candidates.sort(key=lambda x: (-x['score'], x['cand']['candidate_id']))
    
    # ------------------ TELEMETRY STATS SECTION ------------------
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='stat-number'>{len(candidates)}</div>
            <div class='stat-label'>Total Candidates Screened</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='stat-number' style='color:#10b981;'>{len(scored_candidates)}</div>
            <div class='stat-label'>Clean Matching Profiles</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='stat-number' style='color:#ef4444;'>{skipped_anom}</div>
            <div class='stat-label'>Traps & Anomalies Flagged</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='stat-number' style='color:#a855f7;'>{round(sum([c['cand']['profile']['years_of_experience'] for c in scored_candidates]) / max(1, len(scored_candidates)), 1)} yrs</div>
            <div class='stat-label'>Avg Experience (Clean Pool)</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🏆 Ranked Shortlist", "🔍 Deep Profile Vetting", "📊 Talent Pool Insights", "⚖️ Compare Candidates"])
    
    with tab1:
        st.markdown("### Ranked Match Results (Top Fits)")
        
        # Display options
        max_rows = st.slider("Number of top candidates to display", min_value=10, max_value=100, value=25)
        
        # Build DataFrame
        df_list = []
        for idx, item in enumerate(scored_candidates[:max_rows]):
            cand = item['cand']
            profile = cand['profile']
            signals = cand['redrob_signals']
            df_list.append({
                "Rank": idx + 1,
                "ID": cand['candidate_id'],
                "Name": profile['anonymized_name'],
                "Current Title": profile['current_title'],
                "Experience (YoE)": profile['years_of_experience'],
                "Match Score": f"{round(item['score'], 1)}%",
                "Location": profile['location'],
                "Notice (Days)": signals['notice_period_days'],
                "Active": signals['last_active_date'],
                "Vetted Reason": item['reasoning']
            })
            
        df = pd.DataFrame(df_list)
        if not df.empty:
            st.dataframe(
                df,
                column_config={
                    "Match Score": st.column_config.ProgressColumn("Match Score", help="Composite match confidence", format="%s", min_value=0, max_value=100),
                },
                use_container_width=True,
                hide_index=True
            )
            
            # Export button
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export Shortlist (CSV)",
                data=csv_data,
                file_name="candidate_shortlist.csv",
                mime="text/csv"
            )
        else:
            st.warning("No candidates matched the criteria.")
        
    with tab2:
        st.markdown("### Vetting & Profile Analysis")
        if not scored_candidates:
            st.warning("No clean candidates available for detailed profiling.")
        else:
            # Selectbox to choose candidate
            options = [f"Rank {i+1}: {item['cand']['profile']['anonymized_name']} ({item['cand']['candidate_id']})" for i, item in enumerate(scored_candidates)]
            selected_option = st.selectbox("Select a candidate to deep dive:", options)
            selected_idx = options.index(selected_option)
            item = scored_candidates[selected_idx]
            cand = item['cand']
            profile = cand['profile']
            signals = cand['redrob_signals']
            
            st.markdown("---")
            col_p1, col_p2 = st.columns([2, 1])
            
            with col_p1:
                st.markdown(f"## {profile['anonymized_name']} <span style='font-size:1.2rem; color:#9ca3af;'>({cand['candidate_id']})</span>", unsafe_allow_html=True)
                st.markdown(f"#### **{profile['current_title']}** at **{profile['current_company']}**")
                st.markdown(f"📍 {profile['location']}, {profile['country']} | 💼 {profile['current_industry']} | 🏢 Size: {profile['current_company_size']}")
                
                st.markdown("### Recruiter Summary")
                st.markdown(f"*{profile['summary']}*")
                
                # Verified Status Section
                st.markdown("### Profile Verification Status")
                v_email = "<span class='badge-verified'>✓ Email Verified</span>" if signals['verified_email'] else "<span class='badge-unverified'>✗ Email Unverified</span>"
                v_phone = "<span class='badge-verified'>✓ Phone Verified</span>" if signals['verified_phone'] else "<span class='badge-unverified'>✗ Phone Unverified</span>"
                v_link = "<span class='badge-verified'>✓ LinkedIn Connected</span>" if signals['linkedin_connected'] else "<span class='badge-unverified'>✗ LinkedIn Disconnected</span>"
                
                st.markdown(f"{v_email} {v_phone} {v_link}", unsafe_allow_html=True)
                
                # Career History
                st.markdown("### Career History (Timeline)")
                for job in cand.get("career_history", []):
                    end_str = "Present" if job.get("is_current") else job.get("end_date")
                    st.markdown(f"""
                    <div class='timeline-card'>
                        <div class='timeline-title'>{job.get('title')}</div>
                        <div class='timeline-meta'>{job.get('company')} | {job.get('start_date')} to {end_str} ({job.get('duration_months')} months) | Industry: {job.get('industry')} ({job.get('company_size')} employees)</div>
                        <div style='color:#e2e8f0;'>{job.get('description')}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
            with col_p2:
                # Ranking score card
                st.markdown(f"""
                <div style='background-color:#1e293b; padding:2rem; border-radius:12px; border:1px solid #3b82f6; text-align:center; margin-bottom: 1.5rem;'>
                    <div style='font-size:1rem; text-transform:uppercase; color:#9ca3af; letter-spacing:0.1em;'>Recruiter Match Score</div>
                    <div style='font-size:4rem; font-weight:800; color:#3b82f6; margin:0.5rem 0;'>{round(item['score'], 1)}%</div>
                    <div style='color:#e2e8f0; font-size:0.9rem; font-style:italic;'>"{item['reasoning']}"</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Recruiter Copilot section
                st.markdown("### 🤖 Recruiter Copilot (AI-Assisted)")
                
                with st.expander("✉️ Personalized Outreach Email", expanded=True):
                    email_draft = generate_outreach_email(profile, jd_info, item['score'])
                    st.text_area("Copy outreach template:", value=email_draft, height=250)
                    
                with st.expander("❓ Tailored Interview Questions", expanded=False):
                    questions = generate_interview_questions(profile, jd_info)
                    st.markdown("These questions are generated based on the candidate's career gaps and JD requirements:")
                    st.code(questions, language="text")
                
                # Expected Salary and Notice Period
                st.markdown("### Recruitment Signals")
                st.markdown(f"⏰ **Notice Period**: {signals['notice_period_days']} Days")
                st.markdown(f"💰 **Expected Salary**: {signals['expected_salary_range_inr_lpa']['min']} - {signals['expected_salary_range_inr_lpa']['max']} LPA")
                st.markdown(f"💼 **Preferred Mode**: {signals['preferred_work_mode'].title()}")
                st.markdown(f"✈️ **Willing to Relocate**: {'Yes' if signals['willing_to_relocate'] else 'No'}")
                st.markdown(f"🧑‍💻 **GitHub Activity Score**: {signals['github_activity_score'] if signals['github_activity_score'] != -1 else 'N/A'}")
                
                # Skills matching list
                st.markdown("### Candidate Skills Profile")
                for s in cand.get("skills", []):
                    prof_color = "#3b82f6" if s['proficiency'] == "expert" else "#8b5cf6" if s['proficiency'] == "advanced" else "#a855f7" if s['proficiency'] == "intermediate" else "#6b7280"
                    st.markdown(f"""
                    <div style='margin-bottom:0.8rem;'>
                        <div style='display:flex; justify-content:between; font-size:0.85rem; font-weight:600;'>
                            <span style='flex-grow:1;'>{s['name']}</span>
                            <span style='color:{prof_color}; margin-right: 1rem;'>{s['proficiency'].upper()}</span>
                            <span style='color:#9ca3af;'>👍 {s['endorsements']}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
    with tab3:
        st.markdown("### Marketplace Vetting Analytics")
        
        # Convert dataset variables to DataFrames for charts
        clean_df = pd.DataFrame([
            {
                "Name": c['cand']['profile']['anonymized_name'],
                "YoE": c['cand']['profile']['years_of_experience'],
                "Score": c['score'],
                "Notice": c['cand']['redrob_signals']['notice_period_days'],
                "ResponseRate": c['cand']['redrob_signals']['recruiter_response_rate'] * 100,
                "WorkMode": c['cand']['redrob_signals']['preferred_work_mode']
            } for c in scored_candidates
        ])
        
        if not clean_df.empty and HAS_PLOTLY:
            col_c1, col_c2 = st.columns(2)
            
            with col_c1:
                st.markdown("#### Experience vs. Match Score")
                fig_scatter = px.scatter(
                    clean_df, 
                    x="YoE", 
                    y="Score", 
                    color="Notice",
                    hover_name="Name",
                    size="ResponseRate",
                    labels={"YoE": "Years of Experience", "Score": "Match Score (%)", "Notice": "Notice Period (Days)"},
                    template="plotly_dark",
                    color_continuous_scale=px.colors.sequential.Plasma
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
                
            with col_c2:
                st.markdown("#### Excluded Trap & Noise Breakdown")
                trap_data = pd.DataFrame({
                    "Category": ["Clean Fits", "Integrity Traps (Honeypots)", "YoE Out of Scope", "Irrelevant Titles", "Consulting Only", "Primary CV/Speech"],
                    "Count": [len(scored_candidates), skipped_anom, skipped_yoe, skipped_title, skipped_consult, skipped_cv]
                })
                fig_bar = px.bar(
                    trap_data, 
                    x="Category", 
                    y="Count", 
                    color="Category",
                    template="plotly_dark",
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                st.plotly_chart(fig_bar, use_container_width=True)
                
            st.markdown("---")
            col_c3, col_c4 = st.columns(2)
            
            with col_c3:
                st.markdown("#### Preferred Work Mode Distribution (Clean Pool)")
                mode_counts = clean_df["WorkMode"].value_counts().reset_index()
                mode_counts.columns = ["WorkMode", "Count"]
                fig_pie = px.pie(
                    mode_counts, 
                    values="Count", 
                    names="WorkMode",
                    template="plotly_dark",
                    color_discrete_sequence=px.colors.sequential.RdBu
                )
                st.plotly_chart(fig_pie, use_container_width=True)
                
            with col_c4:
                st.markdown("#### Notice Period Distribution (Clean Pool)")
                fig_hist = px.histogram(
                    clean_df, 
                    x="Notice", 
                    nbins=10, 
                    labels={"Notice": "Notice Period (Days)"},
                    template="plotly_dark",
                    color_discrete_sequence=["#a855f7"]
                )
                st.plotly_chart(fig_hist, use_container_width=True)
        else:
            # Fallback to standard Streamlit charts if Plotly isn't loaded
            st.warning("Plotly is loading or not available. Displaying standard charts.")
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                st.markdown("#### Preferred Work Mode Distribution")
                st.bar_chart(clean_df["WorkMode"].value_counts())
            with col_c2:
                st.markdown("#### Notice Period Distribution")
                st.bar_chart(clean_df["Notice"].value_counts())

    with tab4:
        st.markdown("### Compare Candidates Side-by-Side")
        if len(scored_candidates) < 2:
            st.warning("Please upload a pool with at least 2 clean candidates to compare.")
        else:
            # Multi-select to choose candidates
            comp_options = [f"{item['cand']['profile']['anonymized_name']} ({item['cand']['candidate_id']})" for item in scored_candidates]
            selected_comp = st.multiselect(
                "Select up to 3 candidates to compare:",
                comp_options,
                default=comp_options[:2]
            )
            
            if len(selected_comp) > 3:
                st.error("Please select a maximum of 3 candidates.")
            elif len(selected_comp) < 2:
                st.info("Select at least 2 candidates to view comparison.")
            else:
                st.markdown("<br>", unsafe_allow_html=True)
                cols = st.columns(len(selected_comp))
                
                for idx, option in enumerate(selected_comp):
                    c_idx = comp_options.index(option)
                    item = scored_candidates[c_idx]
                    cand = item['cand']
                    profile = cand['profile']
                    signals = cand['redrob_signals']
                    
                    with cols[idx]:
                        # A nice card header
                        st.markdown(f"""
                        <div style='background-color:#1e293b; padding:1.5rem; border-top: 4px solid #3b82f6; border-radius: 8px 8px 0 0; text-align:center;'>
                            <h3 style='margin:0;'>{profile['anonymized_name']}</h3>
                            <p style='color:#9ca3af; margin:5px 0 0 0;'>{profile['current_title']}</p>
                            <h2 style='color:#3b82f6; margin:10px 0 0 0;'>{round(item['score'], 1)}% Match</h2>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Comparison details
                        st.markdown("#### Professional Details")
                        st.write(f"💼 **Experience**: {profile['years_of_experience']} Years")
                        st.write(f"🏢 **Current Company**: {profile['current_company']} ({profile['current_company_size']} employees)")
                        st.write(f"📍 **Location**: {profile['location']}")
                        
                        st.markdown("#### Availability & Signals")
                        st.write(f"⏰ **Notice Period**: {signals['notice_period_days']} Days")
                        st.write(f"💰 **Expected Salary**: {signals['expected_salary_range_inr_lpa']['min']} - {signals['expected_salary_range_inr_lpa']['max']} LPA")
                        st.write(f"💼 **Work Mode**: {signals['preferred_work_mode'].title()}")
                        st.write(f"📞 **Response Rate**: {round(signals['recruiter_response_rate']*100, 1)}%")
                        
                        st.markdown("#### Core Skills")
                        skills_str = ", ".join([s['name'] for s in cand.get("skills", [])[:5]])
                        st.write(f"🛠️ {skills_str}")
                        
                        st.markdown("#### Vetting Summary")
                        st.info(item['reasoning'])
                        
                        st.markdown("---")
