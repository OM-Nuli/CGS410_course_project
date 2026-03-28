import json
import os
import stanza
from conllu import parse_incr

# ==========================================
# SECTION 1: CONFIGURATION
# ==========================================

LANGUAGES = {
    "english":    ("Human/Data_ud/UD-English-EWT/en_ewt-ud-train.conllu",    "en"),
    "chinese":    ("Human/Data_ud/UD-Chinese-GSD/zh_gsd-ud-train.conllu",    "zh"),
    "thai":       ("Human/Data_ud/UD-Thai-TUD/th_tud-ud-train.conllu",        "th"),
    "vietnamese": ("Human/Data_ud/UD-Vietnamese-VTB/vi_vtb-ud-train.conllu",  "vi"),
    "indonesian": ("Human/Data_ud/UD-Indonesian-GSD/id_gsd-ud-train.conllu",  "id"),
    "spanish":    ("Human/Data_ud/UD-Spanish-GSD/es_gsd-ud-train.conllu",     "es"),
}

OUTPUT_DIR = "Human/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# SECTION 2: READ AND CLEAN UD FILE
# ==========================================

def is_punctuation_token(token):
    """
    Returns True if this token is a standalone punctuation mark.
    Uses the Universal POS tag PUNCT which is consistent across
    all UD treebanks regardless of language.
    """
    return token.get("upos") == "PUNCT"


def read_and_clean_conllu(file_path):
    """
    Reads a .conllu file and returns a list of cleaned sentences.
    Cleaning means: remove tokens tagged as PUNCT.

    For Chinese and Thai, tokens are already properly segmented
    in the conllu file even though the raw text has no spaces.
    We join them with spaces so Stanza can process them correctly.

    Only keeps sentences with at least 3 tokens after cleaning.
    """
    cleaned_sentences = []

    with open(file_path, "r", encoding="utf-8") as f:
        for tokenlist in parse_incr(f):
            tokens = []
            for token in tokenlist:
                # Skip multiword tokens (id is a tuple like (1,2))
                if not isinstance(token["id"], int):
                    continue
                # Skip standalone punctuation using Universal POS tag
                if is_punctuation_token(token):
                    continue
                tokens.append(token["form"])

            # Only keep sentences with at least 3 tokens after cleaning
            if len(tokens) >= 3:
                cleaned_sentences.append(" ".join(tokens))

    return cleaned_sentences

# ==========================================
# SECTION 3: STANZA PARSING
# ==========================================

_models = {}

def get_model(lang_code):
    """
    Load and cache a full Stanza pipeline for a language.
    Models are cached so they are only loaded once per run.
    verbose=False keeps output clean.
    No tokenize_pretokenized here — we let Stanza handle its
    own tokenization for accurate dependency parsing.
    """
    if lang_code not in _models:
        print(f"   Loading Stanza model for '{lang_code}'...")
        _models[lang_code] = stanza.Pipeline(
            lang_code,
            processors='tokenize,pos,lemma,depparse',
            verbose=False
        )
    return _models[lang_code]


def compute_dependency_lengths(doc):
    """
    Given a parsed Stanza document, computes average dependency length.
    Dependency length = |word.id - word.head| for all non-ROOT words.
    ROOT is identified by word.head == 0 and is always skipped.
    word.id and word.head are both 1-based in Stanza.
    """
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
    """
    Takes a list of cleaned sentence strings.
    Parses them through Stanza and computes dependency metrics.
    Skips any sentence shorter than 3 tokens after parsing.
    Skips any sentence that causes a parsing error.
    Returns a list of result dictionaries.
    """
    nlp = get_model(lang_code)
    results = []

    for sent in sentences:
        try:
            doc = nlp(sent)
            avg_dep_length = compute_dependency_lengths(doc)
            sentence_length = sum(len(s.words) for s in doc.sentences)

            if sentence_length < 3:
                continue

            results.append({
                "sentence": sent,
                "avg_dep_length": avg_dep_length,
                "sentence_length": sentence_length
            })

        except Exception as e:
            print(f"   Skipping sentence due to error: {e}")
            continue

    return results

# ==========================================
# SECTION 4: MAIN PIPELINE
# ==========================================

def process_language(lang_name, conllu_path, lang_code):
    """
    Full pipeline for one language:
    1. Read and clean UD conllu file (remove PUNCT tokens)
    2. Reparse cleaned sentences through Stanza
    3. Save output JSON to Human/outputs
    """
    print(f"\n{'='*50}")
    print(f"Processing: {lang_name.upper()} ({lang_code})")
    print(f"{'='*50}")

    # Step 1: Read and clean
    print(f"   Reading and cleaning: {conllu_path}")
    sentences = read_and_clean_conllu(conllu_path)
    print(f"   Cleaned sentences: {len(sentences)}")

    # Step 2: Reparse through Stanza
    print(f"   Reparsing through Stanza...")
    results = parse_sentences(sentences, lang_code)
    print(f"   Successfully parsed: {len(results)} sentences")

    # Step 3: Save output
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

    print("\n ALL LANGUAGES DONE")


if __name__ == "__main__":
    main()

