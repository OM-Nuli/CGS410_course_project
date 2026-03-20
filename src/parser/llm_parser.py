import stanza

# Cache loaded models to avoid reloading
_models = {}

def get_model(lang_code):
    """Load and cache Stanza model for a given language"""
    if lang_code not in _models:
        _models[lang_code] = stanza.Pipeline(lang_code, processors='tokenize,pos,lemma,depparse', verbose=False)
    return _models[lang_code]


def process_raw_sentence(sentence, lang_code):
    """
    Parse a raw sentence and compute average dependency length
    """
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