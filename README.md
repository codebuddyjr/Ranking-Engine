# Redrob AI Talent Vetting & Intelligent Ranking System

An AI-powered candidate ranking and vetting system built for the Redrob Challenge. Rather than relying on simple keyword-matching filters (which are easily gamed and miss high-quality fits), this system simulates the judgment of an expert recruiter by **fully understanding profile consistency, career histories, and behavioral availability signals.**

---

## Key Features

1. **Robust Profile Integrity & Honeypot Vetting**:
   - Detects and filters out synthetic anomalies and trap candidates, including salary range contradictions (min > max expected), logical date errors (last active date before signup date), duplicate job histories, and mismatched title-descriptions (e.g. an "Accountant" with a "Software Engineer" description).
2. **Consulting Exclusions & Product Experience Prioritization**:
   - Explicitly filters out candidates who have *only* worked at IT consulting services firms (such as TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini), while retaining candidates with mixed or product-first experience.
3. **Cross-Referenced Skills Vetting**:
   - Evaluates key job description requirements (Embeddings, Vector Search, LLM Fine-Tuning, ML Evaluation) and cross-references them against career descriptions to detect and down-weight "keyword stuffers".
4. **Behavioral Availability Weighting**:
   - Scores candidates based on their notice periods, login activity cadence, and recruiter response rates to prioritize active, available talent.
5. **Interactive Recruiter Dashboard**:
   - A premium, dark-themed Streamlit user interface that handles candidate uploads (JSON/JSONL), interactive ranking leaderboards, verified status badges, vertical career timeline cards, and pool analytics.

---

## Quick Start & Reproduction

### 1. Requirements
Ensure you have Python 3.10+ installed. Install the minimal dependencies:
```bash
pip install -r requirements.txt
```

### 2. Candidate Ranking Command (Stage 3 Reproduction)
To generate the ranked candidate CSV (`submission.csv`) from the candidate database, run the following command. The script streams the file line-by-line and processes 100,000 candidates in **~1.1 minutes** on CPU:
```bash
python rank.py --candidates ./[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl --out ./submission.csv
```

*Note: The script automatically handles gzip compressed files (e.g., `./candidates.jsonl.gz`) if passed.*

### 3. Run the Interactive Dashboard Locally
To start the recruiter-facing analytics and Deep Vetting dashboard:
```bash
streamlit run app.py
```

---

## Vetting & Scoring Methodology

The composite score (0-100%) is calculated using a multi-weighted scoring system across four core vectors:
- **Technical Skills Alignment (35%)**: Evaluates core domain requirements (Embeddings, Vector Search, Evaluation, Fine-Tuning). Partial weight is given if skills are present in history but not listed, and points are subtracted if listed skills are not mentioned in career histories (preventing keyword stuffing).
- **Position & Experience Fit (25%)**: Prioritizes candidates in the 5-9 YoE sweet spot (experience score scales down for junior or Staff/Principal profiles to match founding team requirements) and checks title relevance.
- **Location Preference (15%)**: Prioritizes local candidates (Noida, Pune, NCR) and India-based talent. International applicants are down-weighted due to lack of visa sponsorship.
- **Behavioral & Availability Signals (25%)**: Incentivizes active seekers (open-to-work flag, recent active logins, high response rates) and penalizes long notice periods (above 60 days).

---

## Hosted Sandbox Deployment

The Streamlit app is designed to be fully self-contained and ready for hosted deployment.

### Streamlit Cloud / HuggingFace Spaces Deployment:
1. Push this workspace code to a GitHub repository.
2. Log in to [Streamlit Community Cloud](https://share.streamlit.io/) or [HuggingFace Spaces](https://huggingface.co/spaces).
3. Connect your repository and select `app.py` as the main entrypoint.
4. Set the build environment. The app will build and run on standard free-tier CPU spaces.
5. Upload a small candidate sample (e.g., `sample_candidates.json`) in the UI sidebar to verify ranking end-to-end.
