import random
import spacy

# Load spaCy model once (efficient)
nlp = spacy.load("en_core_web_sm")


def shuffle_sentence(sentence):
    """
    Randomly shuffle words in a sentence.
    Ensures the shuffled sentence is NOT identical to original.
    """

    words = sentence.split()
    original_words = words.copy()

    # Keep shuffling until order changes (important for short sentences)
    while words == original_words:
        random.shuffle(words)

    return " ".join(words)


def process_random_sentence(sentence):
    """
    Shuffle sentence → parse → compute dependency length
    """

    # Step 1: shuffle sentence
    shuffled = shuffle_sentence(sentence)

    # Step 2: parse using spaCy
    doc = nlp(shuffled)

    dep_lengths = []

    # Step 3: compute dependency lengths
    for token in doc:
        if token.head.i == token.i:
            continue  # skip ROOT

        distance = abs(token.i - token.head.i)
        dep_lengths.append(distance)

    # Step 4: compute average dependency length
    if len(dep_lengths) > 0:
        avg_dep_length = sum(dep_lengths) / len(dep_lengths)
    else:
        avg_dep_length = 0

    # Step 5: return structured output
    return {
        "sentence": shuffled,  # note: shuffled sentence
        "avg_dep_length": avg_dep_length,
        "sentence_length": len(doc)
    }


def parse_random(sentences):
    """
    Takes a list of sentences and returns random baseline results
    """

    results = []

    for sent in sentences:
        result = process_random_sentence(sent)
        results.append(result)

    return results