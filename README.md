Perfect! Here’s a **full polished README** for your repo, integrating everything — project overview, methodology, Python version note, sample output, and usage instructions. It’s clear, concise, and reader-friendly.

---

# CGS410 Course Project – Dependency Length Minimization in LLMs

## ⭐ Project Idea

**Research Question:**
Do large language models (LLMs) generate sentences that obey **Dependency Length Minimization (DLM)** like human languages?

Human languages tend to keep related words close together. LLMs are trained only to predict the next word. We investigate if their sentences follow the same structural efficiency principles.

---

## 📚 Method

1. **Human Sentences (UD Treebank)**

   * Extract dependency parses from Universal Dependencies (UD) datasets.
   * Compute **dependency lengths** for each sentence.

2. **LLM-Generated Sentences**

   * Generate sentences with GPT-2, LLaMA, or similar.
   * Compute dependency lengths using a dependency parser (spaCy).

3. **Random Baseline**

   * Shuffle words from LLM sentences to create random control.
   * Compute dependency lengths for comparison.

4. **Comparison**

   * Compare average dependency length across:

     * Human corpora
     * LLM-generated sentences
     * Random baseline

5. **Visualization**

   * Plots (to be created by teammate) show:

     * Dependency length vs sentence length
     * LLM vs human vs random
     * Cross-language comparisons

---

## 🛠 Repository Structure

```
CGS410_course_project/
│
├─ data/           # UD datasets (e.g., English EWT)
├─ outputs/        # JSON outputs from parsers
├─ src/
│   └─ parser/
│       ├─ ud_parser.py
│       ├─ llm_parser.py
│       └─ random_parser.py
├─ test_ud.py      # Test UD parser
├─ test_llm.py     # Test LLM parser
├─ test_random.py  # Test Random parser
└─ .gitignore
```

---

## ⚡ Usage

### Python Version

> This project is developed and tested with **Python 3.11**. Using other versions (≥3.14) may cause compatibility issues with **spaCy** and **pydantic**.

### Installation

```bash
# Create virtual environment
python -m venv venv

# Activate venv
# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Running Parsers (Test)

```bash
# Test UD parser
python test_ud.py

# Test LLM parser
python test_llm.py

# Test Random baseline parser
python test_random.py
```

### Output Format

All parsers return a list of dictionaries. Example:

```json
{
  "sentence": "The dog runs",
  "avg_dep_length": 1.0,
  "sentence_length": 3
}
```

Output files are saved in `outputs/` as:

* `ud_data.json` → Human sentences
* `llm_data.json` → LLM-generated sentences
* `random_data.json` → Random baseline


## 📌 References

* [Universal Dependencies (UD) Treebanks](https://universaldependencies.org/)
* Liu, H. (2008). Dependency distance as a metric of language comprehension difficulty. *Journal of Cognitive Science*
* GPT-2, LLaMA models (Hugging Face / respective paper)