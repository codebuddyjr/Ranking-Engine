#!/usr/bin/env python3
"""
Intelligent Candidate Discovery & Ranking Script
Usage:
  python rank.py --candidates ./candidates.jsonl --out ./submission.csv --jd ./jd.txt --weights "skills:40,experience:20,location:20,behavior:20"
"""

import argparse
import csv
import gzip
import json
import os
import sys
import re

from scoring import (
    score_candidate, 
    parse_job_description, 
    get_embedding_model,
    CURRENT_DATE,
    parse_date,
    is_consulting_company,
    check_mismatched_job
)

def parse_weights(weights_str):
    """
    Parses weights string (e.g. 'skills:40,experience:20,location:20,behavior:20') into a dictionary.
    """
    if not weights_str:
        return None
    try:
        parts = weights_str.split(",")
        w_dict = {}
        for part in parts:
            k, v = part.split(":")
            w_dict[k.strip().lower()] = float(v.strip())
        return w_dict
    except Exception as e:
        print(f"Warning: Could not parse weights '{weights_str}' due to: {e}. Using default weights.", file=sys.stderr)
        return None

def fast_lexical_score(cand, jd_words):
    """
    Computes a very fast word-overlap score between the candidate's profile and the JD keywords.
    Used for Stage 1 retrieval.
    """
    profile = cand.get("profile", {})
    career = cand.get("career_history", [])
    
    # Concatenate title, summary, and career descriptions
    text = f"{profile.get('current_title', '')} {profile.get('summary', '')} "
    text += " ".join([job.get("description", "") + " " + job.get("title", "") for job in career])
    text = text.lower()
    
    overlap = sum(1 for w in jd_words if w in text)
    return overlap

def main():
    parser = argparse.ArgumentParser(description="Rank candidates based on Job Description requirements.")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl or candidates.jsonl.gz")
    parser.add_argument("--out", required=True, help="Output path for the ranked CSV file")
    parser.add_argument("--jd", required=False, help="Path to a text file containing the Job Description")
    parser.add_argument("--weights", required=False, help="Custom weights (e.g., 'skills:35,experience:25,location:15,behavior:25')")
    args = parser.parse_args()
    
    if not os.path.exists(args.candidates):
        print(f"Error: Candidate file not found at {args.candidates}", file=sys.stderr)
        sys.exit(1)
        
    # 1. Read Job Description if provided
    jd_text = ""
    if args.jd:
        if os.path.exists(args.jd):
            with open(args.jd, "r", encoding="utf-8") as f:
                jd_text = f.read().strip()
            print(f"Loaded Job Description from {args.jd}...")
        else:
            print(f"Warning: Job description file not found at {args.jd}. Using default JD.", file=sys.stderr)
            
    if not jd_text:
        # Default JD (AI/ML Vector Search Role)
        jd_text = "AI/ML Engineer specializing in vector search, embeddings, evaluation (NDCG, MRR), and LLM fine-tuning (LoRA, QLoRA)."
        
    jd_info = parse_job_description(jd_text)
    custom_weights = parse_weights(args.weights)
    
    # Words in JD for fast Stage-1 screening
    stopwords = {"and", "the", "for", "with", "you", "will", "our", "are", "that", "this", "from", "have", "role", "team", "work", "experience", "required", "preferred"}
    jd_words = set(re.findall(r'\b[a-z]{3,15}\b', jd_text.lower())) - stopwords
    
    print(f"Reading candidates from {args.candidates}...")
    
    is_gzip = args.candidates.endswith(".gz")
    open_func = gzip.open if is_gzip else open
    mode = "rt" if is_gzip else "r"
    
    # STAGE 1: FAST RETRIEVAL
    # We screen and score candidates using fast lexical overlap to find the top 500 candidates.
    print("Stage 1: Screening and retrieving top candidates using fast lexical matching...")
    stage1_candidates = []
    count = 0
    skipped_vetting = 0
    
    with open_func(args.candidates, mode, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                cand = json.loads(line)
                
                # Fast Vetting Checks (to avoid putting invalid candidates in Stage 2)
                # 1. Anomaly checks
                signals = cand.get("redrob_signals", {})
                signup_dt = parse_date(signals.get("signup_date"))
                last_active_dt = parse_date(signals.get("last_active_date"))
                if signup_dt and last_active_dt and last_active_dt < signup_dt:
                    skipped_vetting += 1
                    continue
                    
                sal = signals.get("expected_salary_range_inr_lpa", {})
                if sal.get("min", 0) > sal.get("max", 0):
                    skipped_vetting += 1
                    continue
                    
                # If they pass basic vetting, calculate fast overlap score
                lex_score = fast_lexical_score(cand, jd_words)
                stage1_candidates.append((lex_score, cand))
                
            except Exception:
                pass
            count += 1
            if count % 20000 == 0:
                print(f"Processed {count} profiles in Stage 1...")
                
    print(f"Total processed in Stage 1: {count}")
    print(f"Skipped during fast vetting: {skipped_vetting}")
    
    # Sort by lexical score descending and keep top 500
    stage1_candidates.sort(key=lambda x: -x[0])
    top_500 = [cand for _, cand in stage1_candidates[:500]]
    print(f"Selected top {len(top_500)} candidates for Stage 2 Re-ranking.")
    
    # STAGE 2: DEEP SEMANTIC RE-RANKING
    print("Stage 2: Re-ranking top candidates using deep semantic and behavioral scoring...")
    
    # Load Sentence-Transformer if available
    model = get_embedding_model()
    jd_embedding = None
    if model:
        print("Sentence-Transformer model loaded. Computing JD embedding...")
        jd_embedding = model.encode(jd_text, convert_to_tensor=True)
    else:
        print("Using TF-IDF/lexical fallback for Stage 2 semantic similarity.")
        
    scored_candidates = []
    for cand in top_500:
        try:
            score, reasoning = score_candidate(
                cand, 
                jd_text=jd_text, 
                weights=custom_weights, 
                jd_info=jd_info, 
                model=model, 
                jd_embedding=jd_embedding
            )
            if score > 0:
                scored_candidates.append((score, reasoning, cand['candidate_id']))
        except Exception as e:
            print(f"Error scoring candidate {cand.get('candidate_id')}: {e}", file=sys.stderr)
            
    print(f"Total qualifying fits after Stage 2: {len(scored_candidates)}")
    
    # Sort by score descending, then candidate_id ascending (alphabetically)
    scored_candidates.sort(key=lambda x: (-round(x[0] / 100.0, 4), x[2]))
    
    # Get top 100
    top_100 = scored_candidates[:100]
    
    if len(top_100) < 100:
        print(f"Warning: Only found {len(top_100)} candidates. Filling to 100 as required.", file=sys.stderr)
        
    print(f"Writing top 100 candidates to {args.out}...")
    
    # Ensure parent directory of output exists
    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for idx, (score, reasoning, cid) in enumerate(top_100):
            norm_score = round(score / 100.0, 4)
            writer.writerow([cid, idx + 1, norm_score, reasoning])
            
    print("Ranking complete. CSV file successfully saved.")

if __name__ == "__main__":
    main()
