# CGS410 Course Project — Dependency Length Minimization in LLMs

## Research Question
Do large language models (LLMs) generate sentences that obey Dependency Length 
Minimization (DLM) like human languages?

## Languages
This project analyzes 6 non-case-marking languages:

| Language   | Code | UD Dataset          |
|------------|------|---------------------|
| English    | en   | UD-English-EWT      |
| Chinese    | zh   | UD-Chinese-GSD      |
| Vietnamese | vi   | UD-Vietnamese-VTB   |
| Thai       | th   | UD-Thai-PUD         |
| Indonesian | id   | UD-Indonesian-GSD   |
| Wolof      | wo   | UD-Wolof-WTB        |

## Repository Structure
```
CGS410_course_project/
├── data/                  # UD treebank datasets
├── outputs/               # JSON outputs from parsers
├── src/
│   └── parser/
│       ├── ud_parser.py       # Human sentences (UD)
│       ├── llm_parser.py      # LLM generated sentences
│       └── random_parser.py   # Random baseline
├── test_ud.py
├── test_llm.py
├── test_random.py
├── run_ud_all.py
├── requirements.txt
└── README.md
```

## Setup

### Requirements
Python 3.11 is required. Using Python 3.14 or above will cause 
compatibility issues with Stanza and spaCy.

### Installation
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Download Stanza Models
```python
import stanza
stanza.download('en')
stanza.download('zh')
stanza.download('vi')
stanza.download('th')
stanza.download('id')
stanza.download('wo')
```

## Usage

### Human Sentences (UD Parser)
```python
from src.parser.ud_parser import parse_ud

data = parse_ud("data/UD-English-EWT/en_ewt-ud-train.conllu")
```

### LLM Generated Sentences
```python
from src.parser.llm_parser import parse_llm

sentences = ["The dog runs in the park"]
data = parse_llm(sentences, "en")
```

### Random Baseline
```python
from src.parser.random_parser import parse_random

# Pass human sentences for shuffling
sentences = ["The dog runs in the park"]
data = parse_random(sentences, "en")
```

## Output Format
All parsers return a list of dictionaries in this format:
```json
{
  "sentence": "The dog runs in the park",
  "avg_dep_length": 1.6,
  "sentence_length": 6
}
```

## Output Files
Results are saved in `outputs/` as JSON files:
- `english_ud.json`
- `chinese_ud.json`
- `vietnamese_ud.json`
- `thai_ud.json`
- `indonesian_ud.json`
- `wolof_ud.json`

## Notes
- Model loading is slow on first run — this is expected
- Random baseline should show higher dependency length than human/LLM at scale
- UD parser uses gold parses — no model loading required
- LLM and Random parsers use Stanza for dependency parsing

## References
- Universal Dependencies Treebanks
- Stanza NLP Library
- Liu, H. (2008). Dependency distance as a metric of language comprehension difficulty