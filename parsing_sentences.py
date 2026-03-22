#this code is for kaggle
import os
import json
import pandas as pd
import stanza


_models = {}
_SUPPORTED_LANGUAGES = {"en", "zh", "vi", "th", "id", "wo"}

def get_model(lang_code):
    """Load and cache Stanza model for a given language."""
    if not isinstance(lang_code, str) or not lang_code:
        raise ValueError("lang_code must be a non-empty string")

    if lang_code not in _SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language code '{lang_code}'. "
            f"Supported: {', '.join(sorted(_SUPPORTED_LANGUAGES))}"
        )

    if lang_code not in _models:
       
        processors_list = "tokenize,pos,lemma,depparse"
        try:
            _models[lang_code] = stanza.Pipeline(
                lang_code,
                processors=processors_list,
                verbose=False,
            )
        except Exception:
           
            stanza.download(lang_code)
            _models[lang_code] = stanza.Pipeline(
                lang_code,
                processors=processors_list,
                verbose=False,
            )

    return _models[lang_code]

def process_raw_sentence(sentence, lang_code):
    """Parse a raw sentence and compute average dependency length."""
    if not isinstance(sentence, str):
        raise ValueError("sentence must be a string")

    normalized = sentence.strip()
    if not normalized:
        return {
            "sentence": sentence,
            "avg_dep_length": 0,
            "sentence_length": 0,
        }

    nlp = get_model(lang_code)
    doc = nlp(sentence)

    dep_lengths = []

    for sent in doc.sentences:
        for word in sent.words:
           
            if word.head == 0:
                continue

            
            distance = abs(word.id - word.head)
            dep_lengths.append(distance)

    if len(dep_lengths) > 0:
        avg_dep_length = sum(dep_lengths) / len(dep_lengths)
    else:
        avg_dep_length = 0

    return {
        "sentence": sentence,
        "avg_dep_length": avg_dep_length,
        "sentence_length": sum(len(s.words) for s in doc.sentences)
    }

def parse_llm(sentences, lang_code):
    """
    Takes a list of sentences and a language code
    Returns processed dependency metrics
    """
    results = []
    for sent in sentences:
        result = process_raw_sentence(sent, lang_code)
        results.append(result)
    return results





INPUT_BASE = "/kaggle/input"
INPUT_CSV_DIR = None


for root, dirs, files in os.walk(INPUT_BASE):
    if 'outputs2' in dirs:
        INPUT_CSV_DIR = os.path.join(root, 'outputs2')
        break

if not INPUT_CSV_DIR:
    raise FileNotFoundError("Could not find the 'outputs2' folder! Please check the right-hand panel to ensure your dataset is attached.")

OUTPUT_DIR = "/kaggle/working/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

languages = {
    "english": "en", "chinese": "zh", "vietnamese": "vi", 
    "thai": "th", "indonesian": "id", "wolof": "wo"
}

print(f"Found your files! Reading CSVs from: {INPUT_CSV_DIR}")
print(f"Saving JSONs to: {OUTPUT_DIR}\n")

for lang_name, lang_code in languages.items():
    print(f"{'-'*40}\nProcessing: {lang_name.upper()}\n{'-'*40}")
    
    csv_path = os.path.join(INPUT_CSV_DIR, f"{lang_name}_generated.csv")
    out_path = os.path.join(OUTPUT_DIR, f"{lang_name}_llm.json")
    
    if os.path.exists(csv_path):
       
        df = pd.read_csv(csv_path)
        sentences = df.iloc[:, 0].dropna().astype(str).tolist()
        print(f"Found {len(sentences)} sentences. Parsing...")
        
  
        raw_data = parse_llm(sentences, lang_code)
            
      
        cleaned_data = [d for d in raw_data if d["sentence_length"] > 2 and d["avg_dep_length"] > 0]
        
        
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(cleaned_data, f, indent=4, ensure_ascii=False)
            
        print(f"✅ Saved {len(cleaned_data)} valid parses to {out_path}\n")
    else:
        print(f"⚠️ File not found: {csv_path}\n")

print("🎉 All LLM parses complete! Check the /kaggle/working/outputs/ folder.")