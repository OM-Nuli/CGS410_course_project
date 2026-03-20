import random
import stanza

# Cache loaded models to avoid reloading
_models = {}

def get_model(lang_code):
    """Load and cache Stanza model for a given language"""
    if lang_code not in _models:
        _models[lang_code] = stanza.Pipeline(lang_code, processors='tokenize,pos,lemma,depparse', verbose=False)
    return _models[lang_code]


def shuffle_sentence(sentence, lang_code):
    """
    Tokenize using Stanza first, then shuffle tokens.
    Handles languages without spaces (Chinese, Thai, etc.)
    """
    nlp = get_model(lang_code)
    doc = nlp(sentence)

    # Extract tokens using Stanza tokenizer
    words = [word.text for sent in doc.sentences for word in sent.words]
    original = words.copy()

    # Keep shuffling until order actually changes
    while words == original:
        random.shuffle(words)

    return " ".join(words)


def process_random_sentence(sentence, lang_code):
    """
    Takes a human sentence, shuffles it, parses it,
    and computes average dependency length.
    This serves as the random (no structure) baseline.
    """
    # Step 1: shuffle word order using Stanza tokenizer
    shuffled = shuffle_sentence(sentence, lang_code)

    # Step 2: parse shuffled sentence with Stanza
    nlp = get_model(lang_code)
    doc = nlp(shuffled)

    dep_lengths = []

    for sent in doc.sentences:
        for word in sent.words:
            # Skip ROOT
            if word.head == 0:
                continue

            distance = abs(word.id - word.head)
            dep_lengths.append(distance)

    if len(dep_lengths) > 0:
        avg_dep_length = sum(dep_lengths) / len(dep_lengths)
    else:
        avg_dep_length = 0

    return {
        "sentence": shuffled,
        "avg_dep_length": avg_dep_length,
        "sentence_length": sum(len(s.words) for s in doc.sentences)
    }


def parse_random(sentences, lang_code):
    """
    Takes a list of human sentences and a language code.
    Returns random baseline dependency metrics.
    """
    results = []

    for sent in sentences:
        result = process_random_sentence(sent, lang_code)
        results.append(result)

    return results