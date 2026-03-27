import stanza

# Cache or download loaded models to avoid reloading
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
        try:
            _models[lang_code] = stanza.Pipeline(
                lang_code,
                processors="tokenize,pos,lemma,depparse",
                verbose=False,
            )
        except Exception:
            # Try download if model is missing
            stanza.download(lang_code)
            _models[lang_code] = stanza.Pipeline(
                lang_code,
                processors="tokenize,pos,lemma,depparse",
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
            # Skip ROOT (head is 0 in Stanza)
            if word.head == 0:
                continue

            # word.id and word.head are 1-based in Stanza
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