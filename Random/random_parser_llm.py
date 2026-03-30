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

# Languages that don't use spaces between words
NO_SPACE_LANGS = {"zh", "th"}

OUTPUT_DIR = "/kaggle/working/LLM/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output dir ready: {OUTPUT_DIR}")
print(f"Dir exists: {os.path.exists(OUTPUT_DIR)}")

def read_llm_sentences(file_path, lang_code):
    cleaned_sentences = []
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for entry in data:
        sent = entry.get("sentence", "").strip()
        if not sent:
            continue
        # Chinese/Thai: split by character so shuffle works token-by-token
        # All others: split by space
        if lang_code in NO_SPACE_LANGS:
            tokens = list(sent)
        else:
            tokens = sent.split(" ")
        if len(tokens) >= 3:
            cleaned_sentences.append(tokens)
    return cleaned_sentences

def shuffle_sentences(sentences):
    shuffled = []
    for tokens in sentences:
        tokens_copy = tokens[:]
        random.shuffle(tokens_copy)
        shuffled.append(" ".join(tokens_copy))
    return shuffled

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

def compute_dependency_lengths(doc):
    dep_lengths = []
    for sent in doc.sentences:
        for word in sent.words:
            if word.head == 0:
                continue
            distance = abs(word.id - word.head)
            dep_lengths.append(distance)
    if len(dep_lengths) == 0:
        return 0
    return sum(dep_lengths) / len(dep_lengths)

def parse_sentences(sentences, lang_code):
    nlp = get_model(lang_code)
    results = []

    batch_size = 64
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        try:
            docs = nlp.bulk_process(batch)
            for sent_text, doc in zip(batch, docs):
                avg_dep_length = compute_dependency_lengths(doc)
                sentence_length = sum(len(s.words) for s in doc.sentences)
                if sentence_length < 3:
                    continue
                results.append({
                    "sentence": sent_text,
                    "avg_dep_length": avg_dep_length,
                    "sentence_length": sentence_length
                })
        except Exception as e:
            print(f"   Skipping batch due to error: {e}")
            continue

        if i % 500 == 0:
            print(f"   Processed {i}/{len(sentences)} sentences...")

    return results

def process_language(lang_name, json_path, lang_code):
    print(f"\n{'='*50}")
    print(f"Processing: {lang_name.upper()} ({lang_code})")
    print(f"{'='*50}")

    if not os.path.exists(json_path):
        print(f"   WARNING: File not found — {json_path}. Skipping.")
        return

    print(f"   Reading and cleaning: {json_path}")
    sentences = read_llm_sentences(json_path, lang_code)
    print(f"   Cleaned sentences: {len(sentences)}")

    print(f"   Randomly shuffling words in each sentence...")
    shuffled = shuffle_sentences(sentences)
    print(f"   Shuffled sentences: {len(shuffled)}")

    print(f"   Reparsing through Stanza...")
    results = parse_sentences(shuffled, lang_code)
    print(f"   Successfully parsed: {len(results)} sentences")

    output_path = os.path.join(OUTPUT_DIR, f"{lang_name}_random_llm.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    # Verify save
    if os.path.exists(output_path):
        size = os.path.getsize(output_path)
        print(f"   SAVED OK: {output_path} ({size} bytes)")
    else:
        print(f"   ERROR: File was NOT saved — {output_path}")

def main():
    for lang_name, (json_path, lang_code) in LANGUAGES.items():
        process_language(lang_name, json_path, lang_code)

    print("\n" + "="*50)
    print("ALL LANGUAGES DONE")
    print("Files in output dir:")
    for f in os.listdir(OUTPUT_DIR):
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
        print(f"   {f} ({size} bytes)")

main()