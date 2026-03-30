from groq import Groq
import json
import time
import re
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

API_KEY = "YOUR_GROQ_KEY_HERE"
OUTPUT_DIR = Path("/content/LLM/sentences")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_PER_LANGUAGE = 2500
SENTENCES_PER_CALL  = 30
DELAY_BETWEEN_CALLS = 5  # Groq free tier: ~30 RPM

# ─────────────────────────────────────────────
# LANGUAGES
# ─────────────────────────────────────────────

LANGUAGES = {
    "en": {"name": "English",    "extra": ""},
    "zh": {"name": "Chinese",    "extra": "Write in Simplified Chinese (简体中文)."},
    "th": {"name": "Thai",       "extra": "Write in Thai script."},
    "vi": {"name": "Vietnamese", "extra": "Write in Vietnamese with correct diacritics."},
    "id": {"name": "Indonesian", "extra": "Use standard Indonesian (Bahasa Indonesia baku)."},
    "es": {"name": "Spanish",    "extra": "Write in standard Spanish."},
}

TOPICS = [
    "daily life and routines",
    "food and cooking",
    "education and schools",
    "transportation and travel",
    "technology and smartphones",
    "health and medicine",
    "nature and weather",
    "work and employment",
    "family and relationships",
    "cities and urban life",
]

SYSTEM_PROMPT = """You are a linguistically diverse sentence generator. Produce natural, fluent sentences that resemble text found in Wikipedia articles, news reports, and blog posts.

Rules:
- Each sentence must be grammatically complete and stand alone.
- Mix simple, compound, and complex sentences with subordinate clauses.
- Aim for 8–20 words per sentence.
- Do NOT translate from English. Write naturally in the target language.
- Do NOT number the sentences.
- Do NOT add bullet points, dashes, or any prefixes.
- Output ONLY the sentences, one per line, with no commentary or blank lines."""

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

client = Groq(api_key=API_KEY)

def clean_sentences(raw_text):
    lines = raw_text.strip().split("\n")
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^[\d]+[\.\)]\s*", "", line)
        line = re.sub(r"^[-•]\s*", "", line)
        line = line.strip()
        if len(line) > 5:
            cleaned.append(line)
    return cleaned

def generate_sentences(lang_name, topic, extra, n):
    extra_note = f"\n{extra}" if extra else ""
    prompt = (
        f"Generate exactly {n} natural {lang_name} sentences about: {topic}.{extra_note}\n\n"
        f"Mix everyday informational writing styles — news, Wikipedia, blog posts. "
        f"Vary sentence structures. Write entirely in {lang_name}. "
        f"One sentence per line, no numbering, no extra text."
    )
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt}
            ]
        )
        return clean_sentences(response.choices[0].message.content)
    except Exception as e:
        print(f"    ⚠️  Error: {e}")
        return []

# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────

def generate_for_language(lang_code):
    config      = LANGUAGES[lang_code]
    lang_name   = config["name"]
    extra       = config["extra"]
    output_path = OUTPUT_DIR / f"{lang_code}_llm_sentences.json"

    # Resume if partially done
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            all_sentences = json.load(f)
        print(f"\n▶ {lang_name}: resuming from {len(all_sentences)} sentences")
    else:
        all_sentences = []
        print(f"\n▶ {lang_name}: starting fresh")

    topic_target = TARGET_PER_LANGUAGE // len(TOPICS)

    for topic in TOPICS:
        topic_count = sum(1 for s in all_sentences if s.get("topic") == topic)

        if topic_count >= topic_target:
            print(f"  ✅ [{topic}] already done ({topic_count} sentences)")
            continue

        calls_needed = max(1, (topic_target - topic_count) // SENTENCES_PER_CALL + 1)
        print(f"  📝 [{topic}] need ~{topic_target - topic_count} more → {calls_needed} call(s)")

        for call_i in range(calls_needed):
            current = sum(1 for s in all_sentences if s.get("topic") == topic)
            if current >= topic_target:
                break

            sentences = generate_sentences(lang_name, topic, extra, SENTENCES_PER_CALL)
            print(f"     call {call_i+1}: got {len(sentences)} sentences")

            for sent in sentences:
                all_sentences.append({
                    "language": lang_code,
                    "topic": topic,
                    "sentence": sent,
                })

            # Save after every call (crash-safe)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(all_sentences, f, ensure_ascii=False, indent=2)

            time.sleep(DELAY_BETWEEN_CALLS)

    print(f"  ✅ {lang_name} done — {len(all_sentences)} sentences saved")

# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────

print("=" * 50)
print("  CGS410 LLM Sentence Generator")
print(f"  Target: ~{TARGET_PER_LANGUAGE} per language")
print("=" * 50)

for lang_code in LANGUAGES:
    generate_for_language(lang_code)

print("\n🎉 All done!")
print("\nSentence counts:")
for lang_code in LANGUAGES:
    path = OUTPUT_DIR / f"{lang_code}_llm_sentences.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"  {lang_code}: {len(data)} sentences")