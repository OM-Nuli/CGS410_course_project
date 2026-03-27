import json
from src.parser.ud_parser import parse_ud

# -------- Language file paths --------
datasets = {
    "english": "data/UD-English-EWT/en_ewt-ud-train.conllu",
    "chinese": "data/UD-Chinese-GSD/zh_gsd-ud-train.conllu",
    "vietnamese": "data/UD-Vietnamese-VTB/vi_vtb-ud-train.conllu",
    "thai": "data/UD-Thai-PUD/th_pud-ud-test.conllu",
    "indonesian": "data/UD-Indonesian-GSD/id_gsd-ud-train.conllu",
    "wolof": "data/UD-Wolof-WTB/wo_wtb-ud-train.conllu"
}

# -------- Loop through languages --------
for lang, path in datasets.items():
    print(f"Processing {lang}...")

    data = parse_ud(path)

    # Save output
    output_path = f"outputs/{lang}_ud.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"Saved: {output_path}")

print("All languages processed successfully!")