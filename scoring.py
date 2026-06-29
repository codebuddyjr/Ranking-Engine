import re
from datetime import datetime

CURRENT_DATE = datetime(2026, 6, 14)

# Lazy loading of Sentence-Transformers
SentenceTransformer = None
util = None
try:
    from sentence_transformers import SentenceTransformer as ST, util as ST_util
    SentenceTransformer = ST
    util = ST_util
except ImportError:
    pass

_model_cache = {}

def get_embedding_model(model_name="all-MiniLM-L6-v2"):
    """
    Lazily loads and caches the Sentence-Transformer model.
    """
    if SentenceTransformer is None:
        return None
    if model_name not in _model_cache:
        try:
            # Force CPU execution to ensure compatibility and low memory footprint
            _model_cache[model_name] = SentenceTransformer(model_name, device="cpu")
        except Exception as e:
            print(f"Warning: Could not load embedding model '{model_name}': {e}. Using TF-IDF fallback.")
            _model_cache[model_name] = None
    return _model_cache[model_name]

def parse_date(d_str):
    if not d_str:
        return None
    try:
        return datetime.strptime(d_str, "%Y-%m-%d")
    except:
        return None

def is_consulting_company(name):
    if not name:
        return False
    name = name.lower()
    consulting_firms = [
        "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini", 
        "tata consultancy", "wipro technologies", "infosys technologies", 
        "mphasis", "tech mahindra", "hcl", "deloitte", "pwc", "ey", "kpmg",
        "l&t", "lnt", "mindtree", "ust global", "virtusa", "persistent systems"
    ]
    for firm in consulting_firms:
        if firm in name:
            return True
    return False

def check_mismatched_job(t, d):
    t = t.lower()
    d = d.lower()
    
    # Non-tech titles vs tech descriptions
    is_non_tech_title = any(kw in t for kw in ["accountant", "marketing", "hr", "sales", "support", "graphic", "content writer", "receptionist"])
    is_tech_desc = any(kw in d for kw in [
        "spark", "kafka", "pipeline", "predictive modeling", "nlp pipeline", 
        "recommendation-style", "semantic search", "fine-tuned llama", 
        "ml feature engineering", "deep learning models", "frontend engineering", 
        "android mobile development", "cloud infrastructure", "test automation",
        "pyspark", "airflow", "snowflake", "scikit-learn", "tensorflow", "pytorch",
        "neural", "nlp", "kubernetes", "docker", "gcp", "aws", "vector search", 
        "sentence transformers", "fine-tuning llms", "llama-2-7b", "mistral-7b", 
        "qlora", "lora", "bert", "embeddings", "milvus", "weaviate", "qdrant", 
        "faiss", "opensearch", "elasticsearch", "information retrieval", "ranking models", 
        "xgboost", "lightgbm"
    ])
    
    if is_non_tech_title and is_tech_desc:
        return True
        
    # Tech titles vs non-tech descriptions
    is_tech_title = any(kw in t for kw in ["engineer", "ml", "ai", "developer", "scientist", "analyst", "programmer"])
    is_non_tech_desc = any(kw in d for kw in [
        "customer support team lead", "mechanical engineering design role", 
        "content writing and seo", "brand design and creative", 
        "senior accounting role", "marketing leadership", 
        "clinical trial data", "legal operations", "compliance officer"
    ])
    
    if is_tech_title and is_non_tech_desc:
        return True
        
    return False

def sk_name_proper(kw):
    mapping = {
        "embeddings": "embeddings",
        "sentence-transformers": "Sentence Transformers",
        "sentence transformers": "Sentence Transformers",
        "bge": "BGE",
        "e5": "E5",
        "vector db": "Vector DBs",
        "vector search": "Vector Search",
        "pinecone": "Pinecone",
        "weaviate": "Weaviate",
        "qdrant": "Qdrant",
        "milvus": "Milvus",
        "opensearch": "OpenSearch",
        "elasticsearch": "Elasticsearch",
        "faiss": "FAISS",
        "ndcg": "NDCG evaluation",
        "mrr": "MRR evaluation",
        "map": "MAP evaluation",
        "evaluation": "ML evaluation",
        "lora": "LoRA",
        "qlora": "QLoRA",
        "peft": "PEFT",
        "fine-tuning": "fine-tuning",
        "fine-tuning llms": "LLM fine-tuning",
        "llm fine-tuning": "LLM fine-tuning"
    }
    return mapping.get(kw.lower(), kw)

def parse_job_description(jd_text):
    """
    Dynamically extracts required skills, experience level, and target titles from a Job Description.
    """
    if not jd_text or len(jd_text.strip()) < 10:
        # Default to original challenge JD (AI/ML Vector Search Role)
        return {
            "skills": ["embeddings", "vector search", "evaluation", "fine-tuning"],
            "min_yoe": 5,
            "max_yoe": 9,
            "titles": ["ai engineer", "ml engineer", "machine learning engineer", "nlp engineer", "data scientist", "research engineer"]
        }
        
    jd_lower = jd_text.lower()
    
    # 1. Extract skills from a comprehensive vocabulary list
    vocab = [
        "embeddings", "sentence-transformers", "sentence transformers", "bge", "e5",
        "vector db", "vector search", "pinecone", "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch", "faiss", "pgvector",
        "ndcg", "mrr", "map", "evaluation", "ab test", "a/b test",
        "lora", "qlora", "peft", "fine-tuning", "fine-tuning llms", "llm fine-tuning", "sft",
        "learning to rank", "ltr", "ranking models", "re-ranking", "re-ranker", "xgboost", "lightgbm",
        "nlp", "natural language processing", "information retrieval", "ir",
        "python", "pytorch", "tensorflow", "scikit-learn", "keras",
        "kubernetes", "docker", "aws", "gcp", "azure", "ci/cd", "pipeline", "pyspark", "spark", "kafka", "airflow", "snowflake",
        "backend", "frontend", "fullstack", "react", "node", "javascript", "typescript", "java", "golang", "c++", "sql", "nosql",
        "devops", "mlops", "ci/cd", "jenkins", "terraform", "ansible"
    ]
    extracted_skills = []
    for skill in vocab:
        if re.search(r'\b' + re.escape(skill) + r'\b', jd_lower):
            extracted_skills.append(skill)
            
    if not extracted_skills:
        # Fallback to some generic nouns in the text if nothing matched
        words = re.findall(r'\b[a-z]{3,15}\b', jd_lower)
        # Filter out common stop words
        stopwords = {"and", "the", "for", "with", "you", "will", "our", "are", "that", "this", "from", "have", "role", "team", "work", "experience"}
        extracted_skills = list(set([w for w in words if w not in stopwords]) - set(extracted_skills))[:5]
        
    # 2. Extract Years of Experience
    min_yoe = 4
    max_yoe = 12
    # Matches: "5+ years", "5-9 years", "5 years", "at least 5 years", "5+ yoe"
    yoe_matches = re.findall(r'(\d+)\s*(?:-|to)?\s*(\d+)?\s*(?:years?|yrs?|yoe)(?:\s+of)?\s*(?:experience|exp)?', jd_lower)
    if yoe_matches:
        try:
            val1 = int(yoe_matches[0][0])
            if yoe_matches[0][1]:
                val2 = int(yoe_matches[0][1])
                min_yoe = max(0, val1)
                max_yoe = val2
            else:
                min_yoe = max(0, val1)
                max_yoe = min_yoe + 4
        except ValueError:
            pass

    # 3. Extract Titles
    titles_vocab = [
        "machine learning engineer", "ml engineer", "ai engineer", "artificial intelligence engineer",
        "nlp engineer", "data scientist", "research engineer", "backend engineer", "software engineer",
        "frontend engineer", "fullstack engineer", "devops engineer", "data engineer", "qa engineer",
        "product manager", "project manager", "scrum master"
    ]
    extracted_titles = []
    for title in titles_vocab:
        if title in jd_lower:
            extracted_titles.append(title)
            
    if not extracted_titles:
        # Fallback to general terms
        extracted_titles = ["engineer", "developer", "scientist", "analyst", "manager"]
        
    return {
        "skills": extracted_skills[:10], # Cap at top 10 skills
        "min_yoe": min_yoe,
        "max_yoe": max_yoe,
        "titles": extracted_titles
    }

def compute_tfidf_similarity(jd_text, candidate_text):
    """
    Computes TF-IDF cosine similarity between Job Description and Candidate Profile.
    Used as a fast Stage-1 lexical scorer or as a Stage-2 fallback.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf = vectorizer.fit_transform([jd_text, candidate_text])
        return float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
    except Exception:
        # Simple Jaccard similarity fallback if sklearn fails
        jd_words = set(re.findall(r'\b\w+\b', jd_text.lower()))
        cand_words = set(re.findall(r'\b\w+\b', candidate_text.lower()))
        if not jd_words:
            return 0.0
        return len(jd_words.intersection(cand_words)) / len(jd_words)

def compute_fallback_semantic_score(career_text):
    """
    The original rule-based keyword density scoring function for backwards compatibility.
    """
    career_text = career_text.lower()
    words = re.findall(r'\b\w+\b', career_text)
    doc_len = len(words)
    if doc_len == 0:
        return 0.0
        
    term_clusters = [
        (["embeddings", "dense retrieval", "semantic search", "dense vectors", "bi-encoder", "cross-encoder", "dense search"], 2.0),
        (["sentence-transformers", "sentence transformers", "bge", "e5", "mpnet"], 2.0),
        (["vector db", "vector search", "vector database", "hybrid search", "hybrid retrieval"], 2.0),
        (["pinecone", "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch", "faiss", "pgvector"], 2.0),
        (["ndcg", "mrr", "map", "ranking evaluation", "offline evaluation", "evaluation framework", "ab test", "a/b test", "offline-online correlation"], 2.5),
        (["lora", "qlora", "peft", "fine-tuning", "fine-tuned", "sft"], 1.5),
        (["learning to rank", "learning-to-rank", "ltr", "ranking models", "re-ranking", "re-ranker", "xgboost", "lightgbm"], 2.0),
        (["nlp", "natural language processing", "information retrieval", "ir"], 1.5)
    ]
    
    avg_len = 150.0
    k1 = 1.2
    b = 0.75
    total_score = 0.0
    
    for terms, weight in term_clusters:
        freq = 0
        for term in terms:
            freq += career_text.count(term)
        if freq > 0:
            tf_factor = (freq * (k1 + 1)) / (freq + k1 * (1.0 - b + b * (doc_len / avg_len)))
            total_score += weight * tf_factor
            
    return total_score

def generate_reasoning(profile, career, signals, score, skills_found, jd_info):
    yoe = profile.get("years_of_experience", 0)
    title = profile.get("current_title", "AI/ML Engineer")
    company = profile.get("current_company", "")
    
    company_phrase = f" at {company}" if company else ""
    op_hash = sum(ord(c) for c in profile.get("anonymized_name", "")) % 3
    if op_hash == 0:
        opening = f"Strong product-focused {title} with {yoe:.1f} years of experience{company_phrase}."
    elif op_hash == 1:
        opening = f"{yoe:.1f}-year veteran {title}{company_phrase} with a solid track record of production shipping."
    else:
        opening = f"Experienced {title}{company_phrase} ({yoe:.1f} YoE) with demonstrated capability in applied systems."
        
    # Technical fit description
    found_desc = [sk_name_proper(s) for s in skills_found]
    
    if len(found_desc) >= 3:
        tech_clause = f"Directly matches core JD needs in {', '.join(found_desc[:-1])}, and {found_desc[-1]}."
    elif len(found_desc) == 2:
        tech_clause = f"Brings strong hands-on experience in both {found_desc[0]} and {found_desc[1]}."
    elif len(found_desc) == 1:
        tech_clause = f"Well-versed in {found_desc[0]}, though adjacent areas can be picked up quickly."
    else:
        tech_clause = "Brings general software engineering and technical experience."

    # Location, notice period and honest concern checks
    loc = profile.get("location", "").split(",")[0].strip()
    np_days = signals.get("notice_period_days", 90)
    
    concerns = []
    if np_days > 60:
        concerns.append(f"notice period of {np_days}d is longer than preferred")
        
    # Find missing skills from the JD
    jd_skills = jd_info.get("skills", [])
    missing_skills = [s for s in jd_skills if s.lower() not in [sf.lower() for sf in skills_found]]
    if missing_skills:
        concerns.append(f"lacks direct experience in {sk_name_proper(missing_skills[0])}")
            
    loc_phrase = ""
    if loc.lower() in ["noida", "pune", "delhi", "gurgaon", "ncr"]:
        loc_phrase = f"located in {loc} (ideal office location)"
    else:
        loc_phrase = f"{loc}-based"
        if signals.get("willing_to_relocate"):
            loc_phrase += " and willing to relocate"
            
    if concerns:
        concern_str = f" Note: Candidate {', '.join(concerns)}."
    else:
        concern_str = " Fits candidate availability constraints perfectly."
        
    reasoning = f"{opening} {tech_clause} {loc_phrase} with {np_days}d notice.{concern_str}"
    return reasoning

def score_candidate(cand, jd_text=None, weights=None, jd_info=None, model=None, jd_embedding=None):
    """
    Computes a composite score (0-100%) for a candidate based on JD alignment and behavioral signals.
    Supports dynamic weights and JD criteria.
    """
    profile = cand.get("profile", {})
    career = cand.get("career_history", [])
    signals = cand.get("redrob_signals", {})
    skills_list = cand.get("skills", [])
    skills = {s['name'].lower(): s for s in skills_list}
    
    # 1. Parse JD if not provided
    if not jd_info:
        jd_info = parse_job_description(jd_text)
    if not jd_text:
        jd_text = "AI/ML Engineer specializing in vector search, embeddings, evaluation, and LLM fine-tuning."
        
    # Default weights
    if not weights:
        weights = {"skills": 0.35, "experience": 0.25, "location": 0.15, "behavior": 0.25}

    # Normalize weights to sum to 1.0
    total_w = sum(weights.values())
    if total_w > 0:
        weights = {k: v / total_w for k, v in weights.items()}
    
    # 2. HARD FILTER FOR ANOMALIES (HONEYPOTS & TRAPS)
    signup_dt = parse_date(signals.get("signup_date"))
    last_active_dt = parse_date(signals.get("last_active_date"))
    if signup_dt and last_active_dt and last_active_dt < signup_dt:
        return 0.0, "Anomaly: last active date before signup date"
        
    sal = signals.get("expected_salary_range_inr_lpa", {})
    sal_min = sal.get("min", 0)
    sal_max = sal.get("max", 0)
    if sal_min > sal_max:
        return 0.0, "Anomaly: expected salary min exceeds max"
        
    for job in career:
        start_dt = parse_date(job.get("start_date"))
        end_dt = parse_date(job.get("end_date"))
        if start_dt and end_dt and end_dt < start_dt:
            return 0.0, "Anomaly: career history job start date after end date"
        if job.get("duration_months", 0) < 0:
            return 0.0, "Anomaly: career history job has negative duration"
            
    descriptions = [job.get("description", "") for job in career if job.get("description", "")]
    if len(descriptions) != len(set(descriptions)):
        return 0.0, "Anomaly: duplicate job descriptions in career history"
        
    for job in career:
        if check_mismatched_job(job.get("title", ""), job.get("description", "")):
            return 0.0, "Anomaly: career job title-description mismatch"
            
    yoe = profile.get("years_of_experience", 0)
    for job in career:
        if job.get("duration_months", 0) / 12.0 > yoe + 0.1:
            return 0.0, "Anomaly: career history job duration exceeds total years of experience"

    expert_zero_dur_count = sum(1 for s in skills_list if s.get("proficiency") == "expert" and s.get("duration_months", 0) == 0)
    if expert_zero_dur_count >= 5:
        return 0.0, "Anomaly: expert proficiency in multiple skills with 0 duration"
            
    # 3. POSITION & EXPERIENCE SCORE
    target_min = jd_info.get("min_yoe", 4)
    target_max = jd_info.get("max_yoe", 12)
    
    # Absolute bounds (with small buffers)
    if yoe < max(0, target_min - 1) or yoe > (target_max + 6):
        return 0.0, "YoE outside valid engineering range"
        
    # Experience scoring
    if target_min <= yoe <= target_max:
        exp_score = 100
    elif (target_min - 1) <= yoe < target_min:
        exp_score = 70 + (yoe - (target_min - 1)) * 30
    elif target_max < yoe <= (target_max + 3):
        exp_score = 100 - (yoe - target_max) * 10
    else: # target_max + 3 to target_max + 6
        exp_score = 70 - (yoe - (target_max + 3)) * 10
        
    # Current Title Check
    curr_title = profile.get("current_title", "").lower()
    target_titles = jd_info.get("titles", [])
    title_score = 0
    
    if any(title in curr_title for title in target_titles):
        title_score = 100
    elif any(word in curr_title for title in target_titles for word in title.split() if len(word) > 3):
        title_score = 80
    elif any(kw in curr_title for kw in ["engineer", "developer", "scientist", "programmer", "architect"]):
        title_score = 60
    else:
        # Exclude completely non-tech roles if this is a tech JD
        is_tech_jd = any(kw in jd_text.lower() for kw in ["engineer", "developer", "scientist", "programmer", "tech", "architect"])
        if is_tech_jd:
            return 0.0, f"Irrelevant title for technical role: {curr_title}"
        title_score = 40
        
    exp_title_weighted = (exp_score * 0.4 + title_score * 0.6)

    # 4. TECHNICAL & SKILLS MATCHING
    jd_skills = jd_info.get("skills", [])
    skills_found = set()
    skill_points = 0
    
    career_text = " ".join([job.get("description", "").lower() for job in career]) + " " + profile.get("summary", "").lower()
    
    if jd_skills:
        for skill_name in jd_skills:
            skill_name_lower = skill_name.lower()
            # If candidate declared the skill
            if skill_name_lower in skills:
                # Anti-keyword-stuffing: check if skill is mentioned in career descriptions
                if skill_name_lower in career_text or any(w in career_text for w in skill_name_lower.split() if len(w) > 3):
                    skills_found.add(skill_name_lower)
                    skill_points += (100.0 / len(jd_skills))
                else:
                    # Penalized score for declared but unmentioned skill
                    skills_found.add(skill_name_lower)
                    skill_points += (50.0 / len(jd_skills))
            else:
                # Not declared, but present in career history
                if skill_name_lower in career_text or any(w in career_text for w in skill_name_lower.split() if len(w) > 3):
                    skills_found.add(skill_name_lower)
                    skill_points += (60.0 / len(jd_skills))
    else:
        skill_points = 50 # Neutral default if no skills found in JD

    # 5. EXCLUSIONS & RESTRICTIONS (Industry & Sub-Domain Checks)
    # Consulting check
    all_consulting = True
    has_career = False
    for job in career:
        has_career = True
        if not is_consulting_company(job.get("company", "")):
            all_consulting = False
            break
    if has_career and all_consulting:
        return 0.0, "Excluded: exclusively consulting company background"
        
    # Subdomain exclusion (e.g. CV/Speech/Robotics without NLP/IR, only if JD is NLP/IR focused)
    has_nlp_ir_jd = any(kw in jd_text.lower() for kw in ["nlp", "embeddings", "vector", "search", "retrieval", "text", "language", "translation", "llm"])
    if has_nlp_ir_jd:
        has_nlp_ir = len(skills_found.intersection({"embeddings", "vector search", "vector db", "nlp"})) > 0 or any(kw in career_text for kw in ["nlp", "retrieval", "search", "ranking", "recommendation", "information retrieval"])
        has_cv_speech_robot = any(kw in career_text or kw in skills for kw in ["computer vision", "opencv", "image classification", "object detection", "cnn", "speech recognition", "tts", "robotics", "speech to text", "text to speech"])
        if has_cv_speech_robot and not has_nlp_ir:
            return 0.0, "Excluded: primary CV/speech/robotics without NLP/IR"
        
    # Pure academic research exclusion check
    all_research = True
    has_production = False
    prod_kws = ["production", "deploy", "scale", "kubernetes", "docker", "aws", "gcp", "azure", "ci/cd", "pipeline", "infrastructure", "latency", "optimization", "monitoring"]
    for kw in prod_kws:
        if kw in career_text:
            has_production = True
            break
            
    research_keywords = ["academic", "research assistant", "phd", "postdoc", "university", "professor", "fellow", "lab"]
    for job in career:
        title = job.get("title", "").lower()
        company = job.get("company", "").lower()
        is_job_research = any(rk in title or rk in company for rk in research_keywords)
        if not is_job_research:
            all_research = False
            break
            
    if has_career and all_research and not has_production:
        return 0.0, "Excluded: purely academic/research career without production deployment"
        
    # Langchain-only exclusion check (Only if JD mentions AI/ML/LLMs)
    has_ai_jd = any(kw in jd_text.lower() for kw in ["ai", "ml", "machine learning", "llm", "deep learning"])
    if has_ai_jd:
        has_llm_wrappers = any(kw in career_text for kw in ["langchain", "llamaindex", "openai", "gpt"])
        has_classic_ml = any(kw in career_text or kw in skills for kw in ["pytorch", "tensorflow", "scikit-learn", "keras", "xgboost", "lightgbm", "regression", "svm", "random forest", "spacy", "nltk", "fasttext", "bert", "embeddings", "ranking", "retrieval"])
        if has_llm_wrappers and not has_classic_ml:
            return 0.0, "Excluded: LangChain-only experience without underlying ML foundations"
        
    # Title chaser exclusion check
    if len(career) >= 3:
        total_months = sum(job.get("duration_months", 0) for job in career)
        avg_months = total_months / len(career)
        is_high_title = any(t in curr_title for t in ["lead", "staff", "principal", "director", "manager"])
        if avg_months < 18.0 and is_high_title:
            return 0.0, "Excluded: title chaser with short job durations"
        
    # 6. LOCATION SCORE
    country = profile.get("country", "").lower()
    loc = profile.get("location", "").lower()
    
    # Try to extract target locations from JD
    target_locations = []
    for city in ["noida", "pune", "delhi", "gurgaon", "ncr", "mumbai", "bangalore", "bengaluru", "hyderabad", "chennai"]:
        if city in jd_text.lower():
            target_locations.append(city)
            
    location_score = 0
    if country == "india":
        if target_locations:
            if any(tl in loc for tl in target_locations):
                location_score = 100
            elif any(tl in ["delhi", "gurgaon", "noida", "ncr"] for tl in target_locations) and any(l in loc for l in ["delhi", "gurgaon", "noida", "ncr"]):
                location_score = 100  # NCR grouping
            elif "bangalore" in loc or "bengaluru" in loc or "hyderabad" in loc or "mumbai" in loc:
                location_score = 85
            else:
                location_score = 70
        else:
            # Default NCR / Pune preference from original spec
            if "noida" in loc or "pune" in loc or "delhi" in loc or "gurgaon" in loc or "ncr" in loc:
                location_score = 100
            elif "mumbai" in loc or "hyderabad" in loc or "bangalore" in loc or "bengaluru" in loc:
                location_score = 90
            else:
                location_score = 70
    else:
        if signals.get("willing_to_relocate", False):
            location_score = 30
        else:
            location_score = 10
            
    # 7. BEHAVIORAL SIGNALS & AVAILABILITY
    np_days = signals.get("notice_period_days", 90)
    if np_days <= 30:
        np_score = 100
    elif np_days <= 60:
        np_score = 80
    elif np_days <= 90:
        np_score = 40
    else:
        np_score = 10
        
    rr = signals.get("recruiter_response_rate", 0.0)
    rr_score = rr * 100
    
    last_act = parse_date(signals.get("last_active_date"))
    if last_act:
        days_inactive = (CURRENT_DATE - last_act).days
        if days_inactive <= 30:
            act_score = 100
        elif days_inactive <= 90:
            act_score = 80
        elif days_inactive <= 180:
            act_score = 50
        else:
            act_score = 10
    else:
        act_score = 0
        
    otw_boost = 1.1 if signals.get("open_to_work_flag", False) else 1.0
    behavior_weighted = (np_score * 0.4 + rr_score * 0.3 + act_score * 0.3) * otw_boost
    
    # 8. SEMANTIC SIMILARITY MATCHING (DENSE OR SPARSE FALLBACK)
    if model and jd_embedding is not None:
        # Encode candidate profile text
        cand_text = f"{profile.get('current_title', '')} {profile.get('summary', '')} {career_text}"
        try:
            cand_embedding = model.encode(cand_text, convert_to_tensor=True)
            sem_sim = float(util.cos_sim(jd_embedding, cand_embedding)[0][0])
            # Scale cosine similarity from [0.1, 0.6] -> [0, 100]
            sem_score = min(100.0, max(0.0, (sem_sim - 0.1) / 0.5 * 100.0))
        except Exception:
            # Fallback to TF-IDF if model encoding fails
            tfidf_sim = compute_tfidf_similarity(jd_text, cand_text)
            sem_score = min(100.0, tfidf_sim * 100.0 * 1.5)
    else:
        # Fallback to TF-IDF / Heuristic
        cand_text = f"{profile.get('current_title', '')} {profile.get('summary', '')} {career_text}"
        # If it matches the default AI/ML JD exactly, use the original heuristic semantic score
        if "vector search" in jd_text.lower() and "embeddings" in jd_text.lower():
            sem_score_raw = compute_fallback_semantic_score(career_text)
            sem_score = min(100.0, (sem_score_raw / 15.0) * 100.0)
        else:
            tfidf_sim = compute_tfidf_similarity(jd_text, cand_text)
            sem_score = min(100.0, tfidf_sim * 100.0 * 1.5)
            
    skills_weighted = (skill_points * 0.4 + sem_score * 0.6)
    
    # 9. COMPOSITE SCORE
    composite_score = (
        skills_weighted * weights.get("skills", 0.35) +
        exp_title_weighted * weights.get("experience", 0.25) +
        location_score * weights.get("location", 0.15) +
        behavior_weighted * weights.get("behavior", 0.25)
    )
    
    # Cap score between 0 and 100
    composite_score = min(100.0, max(0.0, composite_score))
    
    # 10. GENERATING REASONING
    reasoning = generate_reasoning(profile, career, signals, composite_score, skills_found, jd_info)
    
    return composite_score, reasoning
