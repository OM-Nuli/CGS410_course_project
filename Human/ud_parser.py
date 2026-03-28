import json
import os
import stanza
from conllu import parse_incr

LANGUAGES = {
    "english":    ("/kaggle/input/datasets/omnuli/cgs410-dlm/CGS410_course_project/Human/Data_ud/UD-English-EWT/en_ewt-ud-train.conllu",    "en"),
    "chinese":    ("/kaggle/input/datasets/omnuli/cgs410-dlm/CGS410_course_project/Human/Data_ud/UD-Chinese-GSD/zh_gsd-ud-train.conllu",    "zh"),
    "thai":       ("/kaggle/input/datasets/omnuli/cgs410-dlm/CGS410_course_project/Human/Data_ud/UD-Thai-TUD/th_tud-ud-train.conllu",        "th"),
    "vietnamese": ("/kaggle/input/datasets/omnuli/cgs410-dlm/CGS410_course_project/Human/Data_ud/UD-Vietnamese-VTB/vi_vtb-ud-train.conllu",  "vi"),
    "indonesian": ("/kaggle/input/datasets/omnuli/cgs410-dlm/CGS410_course_project/Human/Data_ud/UD-Indonesian-GSD/id_gsd-ud-train.conllu",  "id"),
    "spanish":    ("/kaggle/input/datasets/omnuli/cgs410-dlm/CGS410_course_project/Human/Data_ud/UD-Spanish-GSD/es_gsd-ud-train.conllu",     "es"),
}

OUTPUT_DIR = "/kaggle/working/Human/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def is_punctuation_token(token):
    return token.get("upos") == "PUNCT"

def read_and_clean_conllu(file_path):
    cleaned_sentences = []
    with open(file_path, "r", encoding="utf-8") as f:
        for tokenlist in parse_incr(f):
            tokens = []
            for token in tokenlist:
                if not isinstance(token["id"], int):
                    continue
                if is_punctuation_token(token):
                    continue
                tokens.append(token["form"])
            if len(tokens) >= 3:
                cleaned_sentences.append(" ".join(tokens))
    return cleaned_sentences

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
    
    # Process in batches of 64
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
        
        if i % 1000 == 0:
            print(f"   Processed {i}/{len(sentences)} sentences...")
    
    return results

def process_language(lang_name, conllu_path, lang_code):
    print(f"\n{'='*50}")
    print(f"Processing: {lang_name.upper()} ({lang_code})")
    print(f"{'='*50}")
    print(f"   Reading and cleaning: {conllu_path}")
    sentences = read_and_clean_conllu(conllu_path)
    print(f"   Cleaned sentences: {len(sentences)}")
    print(f"   Reparsing through Stanza...")
    results = parse_sentences(sentences, lang_code)
    print(f"   Successfully parsed: {len(results)} sentences")
    output_path = os.path.join(OUTPUT_DIR, f"{lang_name}_human.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"   Saved to: {output_path}")

def main():
    for lang_name, (conllu_path, lang_code) in LANGUAGES.items():
        if not os.path.exists(conllu_path):
            print(f"\nWARNING: File not found — {conllu_path}. Skipping.")
            continue
        process_language(lang_name, conllu_path, lang_code)
    print("\nALL LANGUAGES DONE")

main()
