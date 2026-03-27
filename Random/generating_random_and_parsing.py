import os
import sys
import json
import random
import stanza
from tqdm import tqdm  # Gives us a beautiful progress bar

# ==========================================
# PART 1: THE OPTIMIZED PARSER
# ==========================================

_tokenizers = {}
_parsers = {}

def get_models(lang_code):
    """
    Loads TWO models to prevent double-processing:
    1. A lightweight tokenizer (super fast) just to split words for shuffling.
    2. The heavy parser (uses GPU batching) for the final analysis.
    """
    if lang_code not in _tokenizers:
        print(f"Loading models for {lang_code} onto GPU...")
        # We explicitly tell Stanza to use the GPU
        _tokenizers[lang_code] = stanza.Pipeline(lang_code, processors='tokenize', use_gpu=True, verbose=False)
        _parsers[lang_code] = stanza.Pipeline(lang_code, processors='tokenize,pos,lemma,depparse', use_gpu=True, verbose=False)
    return _tokenizers[lang_code], _parsers[lang_code]

def parse_random_batched(sentences, lang_code, batch_size=128):
    """
    Processes sentences in large chunks (batches) to keep the GPU fed.
    """
    tokenizer, parser = get_models(lang_code)
    all_results = []
    
    # 1. SHUFFLE PHASE (Using the fast tokenizer)
    print(f"   Shuffling {len(sentences)} sentences...")
    shuffled_strings = []
    
    for sent in sentences:
        doc = tokenizer(sent)
        words = [word.text for s in doc.sentences for word in s.words]
        original = words.copy()
        
        # THE FIX: Only try to shuffle if there are at least 2 UNIQUE words
        if len(set(words)) > 1:
            while words == original:
                random.shuffle(words)
            
        shuffled_strings.append(" ".join(words))

    # 2. PARSE PHASE (Using the heavy parser with GPU Batching)
    print(f"   Parsing dependencies on GPU (Batch Size: {batch_size})...")
    
    in_docs = [stanza.Document([], text=text) for text in shuffled_strings]
    
    out_docs = []
    for i in tqdm(range(0, len(in_docs), batch_size), desc="Parsing Batches"):
        batch = in_docs[i : i + batch_size]
        parsed_batch = parser(batch) 
        out_docs.extend(parsed_batch)

    # 3. METRICS PHASE (Calculate the Dependency Lengths)
    for original_shuffled_text, doc in zip(shuffled_strings, out_docs):
        dep_lengths = []
        for sent in doc.sentences:
            for word in sent.words:
                if word.head == 0:  
                    continue
                distance = abs(word.id - word.head)
                dep_lengths.append(distance)

        avg_dep_length = sum(dep_lengths) / len(dep_lengths) if len(dep_lengths) > 0 else 0

        all_results.append({
            "sentence": original_shuffled_text,
            "avg_dep_length": avg_dep_length,
            "sentence_length": sum(len(s.words) for s in doc.sentences)
        })

    return all_results

# ==========================================
# PART 2: AUTO-DETECT PATHS & RUN
# ==========================================

def find_input_dir():
    print("🔍 Scanning Kaggle directories for your files...")
    for root, dirs, files in os.walk('/kaggle/input'):
        if 'english_ud.json' in files:
            print(f"🎯 Found the correct folder at: {root}")
            return root
    return None

INPUT_DIR = find_input_dir()
OUTPUT_DIR = "/kaggle/working/"

file_to_lang = {
    "english_ud.json": "en",
    "chinese_ud.json": "zh",
    "indonesian_ud.json": "id",
    "thai_ud.json": "th",
    "vietnamese_ud.json": "vi",
    "wolof_ud.json": "wo"
}

def process_all_languages():
    if not INPUT_DIR:
        print("❌ CRITICAL ERROR: Could not find 'english_ud.json' anywhere in /kaggle/input/.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for filename, lang_code in file_to_lang.items():
        input_path = os.path.join(INPUT_DIR, filename)
        
        if not os.path.exists(input_path):
            continue
            
        print(f"\n========================================")
        print(f"🔄 Processing {filename} ({lang_code})")
        print(f"========================================")
        
        stanza.download(lang_code, verbose=False)

        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        try:
            sentences = [item['sentence'] for item in data]
        except TypeError:
             sentences = data 

        print(f"   Loaded all {len(sentences)} sentences. Starting pipeline...")

        # Call the batched function
        results = parse_random_batched(sentences, lang_code, batch_size=128)

        output_filename = filename.replace("_ud.json", "_random.json")
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
            
        print(f"✅ Saved to: {output_path}")

# Start the process
process_all_languages()