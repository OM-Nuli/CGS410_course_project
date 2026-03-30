import json
import os
import random
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

OUTPUT_DIR = "/kaggle/working/Random/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output dir ready: {OUTPUT_DIR}")
print(f"Dir exists: {os.path.exists(OUTPUT_DIR)}")

NUM_LINEARIZATIONS = 100  # Number of random linearizations per sentence

# ─────────────────────────────────────────────
# READ + CLEAN CONLLU (unchanged)
# ─────────────────────────────────────────────

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

# ─────────────────────────────────────────────
# STANZA MODEL CACHE (unchanged)
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
# DEPENDENCY TREE RANDOMIZATION
# ─────────────────────────────────────────────

def build_children_map(words):
    """Build a map of head -> list of dependent word ids."""
    children = {word.id: [] for word in words}
    root_id = None
    for word in words:
        if word.head == 0:
            root_id = word.id
        else:
            children[word.head].append(word.id)
    return children, root_id

def random_linearize(words, children, root_id):
    """
    Starting from root, recursively collect each node with its dependents
    and shuffle their order randomly. Returns list of word ids in new order.
    """
    id_to_word = {word.id: word for word in words}

    def collect(node_id):
        # Get all dependents of this node
        deps = children[node_id]
        # The group is: this node + all its dependents
        group = [node_id] + deps
        # Shuffle the group randomly
        random.shuffle(group)
        # Now recursively expand each element
        result = []
        for item in group:
            if item == node_id:
                result.append(node_id)
            else:
                result.extend(collect(item))
        return result

    ordered_ids = collect(root_id)
    return ordered_ids

def compute_dep_length_from_order(words, ordered_ids):
    """
    Given a new linear order of word ids, compute average dependency length.
    Position is determined by index in ordered_ids (1-indexed).
    Skip ROOT dependencies.
    """
    # Map word_id -> new position (1-indexed)
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

    if not dep_lengths:
        return 0
    return sum(dep_lengths) / len(dep_lengths)

def avg_dep_length_random_linearizations(words, n=NUM_LINEARIZATIONS):
    """
    Parse the sentence, then produce n random linearizations using
    the dependency tree structure. Return the mean avg_dep_length across all.
    """
    children, root_id = build_children_map(words)
    if root_id is None:
        return 0

    total = 0
    for _ in range(n):
        ordered_ids = random_linearize(words, children, root_id)
        total += compute_dep_length_from_order(words, ordered_ids)

    return total / n

# ─────────────────────────────────────────────
# PARSE SENTENCES
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
                # Use first sentence from doc (each input is one sentence)
                if not doc.sentences:
                    continue
                words = doc.sentences[0].words
                if len(words) < 3:
                    continue

                avg_dep_length = avg_dep_length_random_linearizations(words)
                sentence_length = len(words)

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

# ─────────────────────────────────────────────
# PROCESS ONE LANGUAGE
# ─────────────────────────────────────────────

def process_language(lang_name, conllu_path, lang_code):
    print(f"\n{'='*50}")
    print(f"Processing: {lang_name.upper()} ({lang_code})")
    print(f"{'='*50}")

    if not os.path.exists(conllu_path):
        print(f"   WARNING: File not found — {conllu_path}. Skipping.")
        return

    print(f"   Reading and cleaning: {conllu_path}")
    sentences = read_and_clean_conllu(conllu_path)
    print(f"   Cleaned sentences: {len(sentences)}")

    print(f"   Parsing + computing 100 random linearizations per sentence...")
    results = parse_sentences(sentences, lang_code)
    print(f"   Successfully parsed: {len(results)} sentences")

    output_path = os.path.join(OUTPUT_DIR, f"{lang_name}_random_human.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    # Verify save
    if os.path.exists(output_path):
        size = os.path.getsize(output_path)
        print(f"   SAVED OK: {output_path} ({size} bytes)")
    else:
        print(f"   ERROR: File was NOT saved — {output_path}")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    for lang_name, (conllu_path, lang_code) in LANGUAGES.items():
        process_language(lang_name, conllu_path, lang_code)

    print("\n" + "="*50)
    print("ALL LANGUAGES DONE")
    print("Files in output dir:")
    for f in os.listdir(OUTPUT_DIR):
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
        print(f"   {f} ({size} bytes)")

main()