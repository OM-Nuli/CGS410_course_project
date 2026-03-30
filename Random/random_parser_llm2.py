import json
import os
import random
import stanza

LANGUAGES = {

    "english":    ("/kaggle/input/datasets/omnuli/cgs410-dlm/CGS410_course_project/LLM/LLM_sentences_grok/en_llm_sentences.json", "en"),
    "chinese":    ("/kaggle/input/datasets/omnuli/cgs410-dlm/CGS410_course_project/LLM/LLM_sentences_grok/zh_llm_sentences.json", "zh"),
    "thai":       ("/kaggle/input/datasets/omnuli/cgs410-dlm/CGS410_course_project/LLM/LLM_sentences_grok/th_llm_sentences.json", "th"),
    "vietnamese": ("/kaggle/input/datasets/omnuli/cgs410-dlm/CGS410_course_project/LLM/LLM_sentences_grok/vi_llm_sentences.json", "vi"),
    "indonesian": ("/kaggle/input/datasets/omnuli/cgs410-dlm/CGS410_course_project/LLM/LLM_sentences_grok/id_llm_sentences.json", "id"),
    "spanish":    ("/kaggle/input/datasets/omnuli/cgs410-dlm/CGS410_course_project/LLM/LLM_sentences_grok/es_llm_sentences.json", "es"),
}
OUTPUT_DIR = "/kaggle/working/Random/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output dir ready: {OUTPUT_DIR}")

NUM_LINEARIZATIONS = 100

# ─────────────────────────────────────────────
# READ JSON SENTENCES
# ─────────────────────────────────────────────

def read_json_sentences(file_path, lang_code):
    """
    Read sentences from JSON. For space-free languages (zh, th),
    skip the split()-based length check — just require non-empty string.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    no_space_langs = {"zh", "th"}

    sentences = []
    for item in data:
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict):
            text = (item.get("sentence") or item.get("text") or item.get("content") or "").strip()
        else:
            continue

        if not text:
            continue

        # For space-free languages, trust Stanza tokenizer — don't use split()
        if lang_code in no_space_langs:
            if len(text) >= 3:          # at least 3 characters
                sentences.append(text)
        else:
            if len(text.split()) >= 3:  # space-tokenized word count
                sentences.append(text)

    return sentences

# ─────────────────────────────────────────────
# STANZA MODEL CACHE
# ─────────────────────────────────────────────

_models = {}

def get_model(lang_code):
    if lang_code not in _models:
        print(f"   Loading Stanza model for '{lang_code}'...")
        _models[lang_code] = stanza.Pipeline(
            lang_code,
            processors='tokenize,pos,lemma,depparse',
            tokenize_no_ssplit=True,   # treat each input as a single sentence
            verbose=False,
            use_gpu=False
        )
    return _models[lang_code]

# ─────────────────────────────────────────────
# DEPENDENCY TREE RANDOMIZATION
# ─────────────────────────────────────────────

def build_children_map(words):
    children = {word.id: [] for word in words}
    root_id = None
    for word in words:
        if word.head == 0:
            root_id = word.id
        else:
            children[word.head].append(word.id)
    return children, root_id

def random_linearize(words, children, root_id):
    def collect(node_id):
        deps = children[node_id]
        group = [node_id] + deps
        random.shuffle(group)
        result = []
        for item in group:
            if item == node_id:
                result.append(node_id)
            else:
                result.extend(collect(item))
        return result
    return collect(root_id)

def compute_dep_length_from_order(words, ordered_ids):
    position = {wid: idx + 1 for idx, wid in enumerate(ordered_ids)}
    dep_lengths = []
    for word in words:
        if word.head == 0:
            continue
        new_pos_word = position.get(word.id)
        new_pos_head = position.get(word.head)
        if new_pos_word is None or new_pos_head is None:
            continue
        dep_lengths.append(abs(new_pos_word - new_pos_head))
    return sum(dep_lengths) / len(dep_lengths) if dep_lengths else 0

def avg_dep_length_random_linearizations(words, n=NUM_LINEARIZATIONS):
    children, root_id = build_children_map(words)
    if root_id is None:
        return 0
    total = sum(
        compute_dep_length_from_order(words, random_linearize(words, children, root_id))
        for _ in range(n)
    )
    return total / n

# ─────────────────────────────────────────────
# PARSE SENTENCES
# ─────────────────────────────────────────────

def parse_sentences(sentences, lang_code):
    nlp = get_model(lang_code)
    results = []

    # Smaller batches for Chinese/Thai — segmentation is heavier
    batch_size = 32
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        try:
            docs = nlp.bulk_process(batch)
            for sent_text, doc in zip(batch, docs):
                if not doc.sentences:
                    continue
                words = doc.sentences[0].words

                # Use Stanza-tokenized word count (not split())
                if len(words) < 3:
                    continue

                results.append({
                    "sentence": sent_text,
                    "avg_dep_length": avg_dep_length_random_linearizations(words),
                    "sentence_length": len(words)
                })
        except Exception as e:
            print(f"   Skipping batch due to error: {e}")
            continue

        if i % 200 == 0:
            print(f"   Processed {i}/{len(sentences)} sentences...")

    return results

# ─────────────────────────────────────────────
# PROCESS ONE LANGUAGE
# ─────────────────────────────────────────────

def process_language(lang_name, json_path, lang_code):
    print(f"\n{'='*50}")
    print(f"Processing: {lang_name.upper()} ({lang_code})")
    print(f"{'='*50}")

    if not os.path.exists(json_path):
        print(f"   WARNING: File not found — {json_path}. Skipping.")
        return

    print(f"   Reading JSON: {json_path}")
    sentences = read_json_sentences(json_path, lang_code)
    print(f"   Loaded sentences: {len(sentences)}")

    print(f"   Parsing + computing {NUM_LINEARIZATIONS} random linearizations per sentence...")
    results = parse_sentences(sentences, lang_code)
    print(f"   Successfully parsed: {len(results)} sentences")

    output_path = os.path.join(OUTPUT_DIR, f"{lang_name}_random_llm_grok.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    if os.path.exists(output_path):
        print(f"   SAVED OK: {output_path} ({os.path.getsize(output_path)} bytes)")
    else:
        print(f"   ERROR: File was NOT saved — {output_path}")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    for lang_name, (json_path, lang_code) in LANGUAGES.items():
        process_language(lang_name, json_path, lang_code)

    print("\n" + "="*50)
    print("ALL LANGUAGES DONE")
    print("Files in output dir:")
    for f in os.listdir(OUTPUT_DIR):
        print(f"   {f} ({os.path.getsize(os.path.join(OUTPUT_DIR, f))} bytes)")

main()