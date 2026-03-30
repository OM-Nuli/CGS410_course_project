"""
LLM Sentence Parser — CGS410 DLM Project
Reads generated JSON sentences, parses through Stanza,
computes avg dependency length — same logic as Human parser.

Platform: Kaggle
Input:  /kaggle/input/datasets/omnuli/cgs410-dlm/CGS410_course_project/LLM/sentences/{lang}_llm_sentences.json
Output: /kaggle/working/LLM/outputs/{lang}_llm.json
"""

import json
import os
import stanza

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

LANGUAGES = {
    "english":    ("en", "/kaggle/input/datasets/omnuli/cgs410-dlm/CGS410_course_project/LLM/sentences/en_llm_sentences.json"),
    "chinese":    ("zh", "/kaggle/input/datasets/omnuli/cgs410-dlm/CGS410_course_project/LLM/sentences/zh_llm_sentences.json"),
    "thai":       ("th", "/kaggle/input/datasets/omnuli/cgs410-dlm/CGS410_course_project/LLM/sentences/th_llm_sentences.json"),
    "vietnamese": ("vi", "/kaggle/input/datasets/omnuli/cgs410-dlm/CGS410_course_project/LLM/sentences/vi_llm_sentences.json"),
    "indonesian": ("id", "/kaggle/input/datasets/omnuli/cgs410-dlm/CGS410_course_project/LLM/sentences/id_llm_sentences.json"),
    "spanish":    ("es", "/kaggle/input/datasets/omnuli/cgs410-dlm/CGS410_course_project/LLM/sentences/es_llm_sentences.json"),
}

OUTPUT_DIR = "/kaggle/working/LLM/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# STANZA MODEL CACHE (same as human parser)
# ─────────────────────────────────────────────

_models = {}

def get_model(lang_code):
    if lang_code not in _models:
        print(f"   Loading Stanza model for '{lang_code}'...")
        _models[lang_code] = stanza.Pipeline(
            lang_code,
            processors='tokenize,pos,lemma,depparse',
            verbose=False,
            use_gpu=False
        )
    return _models[lang_code]

# ─────────────────────────────────────────────
# DEPENDENCY LENGTH (same formula as human parser)
# ─────────────────────────────────────────────

def compute_dependency_lengths(doc):
    dep_lengths = []
    for sent in doc.sentences:
        for word in sent.words:
            if word.head == 0:   # skip ROOT
                continue
            distance = abs(word.id - word.head)
            dep_lengths.append(distance)
    if len(dep_lengths) == 0:
        return 0
    return sum(dep_lengths) / len(dep_lengths)

# ─────────────────────────────────────────────
# READ INPUT JSON
# ─────────────────────────────────────────────

def read_llm_sentences(json_path):
    """Read generated sentences from JSON file, return list of sentence strings."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    sentences = [entry["sentence"] for entry in data if entry.get("sentence", "").strip()]
    return sentences

# ─────────────────────────────────────────────
# PARSE SENTENCES (same batch logic as human parser)
# ─────────────────────────────────────────────

def parse_sentences(sentences, lang_code):
    nlp = get_model(lang_code)
    results = []

    batch_size = 64
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        try:
            docs = nlp.bulk_process(batch)
            for sent_text, doc in zip(batch, docs):
                avg_dep_length = compute_dependency_lengths(doc)
                sentence_length = sum(len(s.words) for s in doc.sentences)
                if sentence_length < 3:   # same filter as human parser
                    continue
                results.append({
                    "sentence": sent_text,
                    "avg_dep_length": avg_dep_length,
                    "sentence_length": sentence_length
                })
        except Exception as e:
            print(f"   Skipping batch due to error: {e}")
            continue

        if i % 1000 == 0:
            print(f"   Processed {i}/{len(sentences)} sentences...")

    return results

# ─────────────────────────────────────────────
# PROCESS ONE LANGUAGE
# ─────────────────────────────────────────────

def process_language(lang_name, lang_code, json_path):
    print(f"\n{'='*50}")
    print(f"Processing: {lang_name.upper()} ({lang_code})")
    print(f"{'='*50}")

    print(f"   Reading sentences from: {json_path}")
    sentences = read_llm_sentences(json_path)
    print(f"   Total sentences loaded: {len(sentences)}")

    print(f"   Parsing through Stanza...")
    results = parse_sentences(sentences, lang_code)
    print(f"   Successfully parsed: {len(results)} sentences")

    output_path = os.path.join(OUTPUT_DIR, f"{lang_name}_llm.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"   Saved to: {output_path}")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    for lang_name, (lang_code, json_path) in LANGUAGES.items():
        if not os.path.exists(json_path):
            print(f"\nWARNING: File not found — {json_path}. Skipping.")
            continue
        process_language(lang_name, lang_code, json_path)

    print("\nALL LANGUAGES DONE")

main()