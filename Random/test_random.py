from src.parser.random_parser import parse_random

test_data = {
    "en": ["The dog runs in the park"],
    "zh": ["狗在公园里跑"],
    "vi": ["Con chó chạy trong công viên"],
    "th": ["สุนัขวิ่งในสวนสาธารณะ"],
    "id": ["Anjing berlari di taman"],
    "wo": ["Xaj bi dafa cal ci parc bi"]
}

for lang, sentences in test_data.items():
    print(f"\nLanguage: {lang}")
    results = parse_random(sentences, lang)
    for r in results:
        print(r)
