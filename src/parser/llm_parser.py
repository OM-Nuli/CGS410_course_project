import spacy

# Load the spaCy English model once (efficient — don't reload per sentence)
nlp = spacy.load("en_core_web_sm")


def process_raw_sentence(sentence):
    """
    Takes a raw sentence and computes:
    - average dependency length
    - sentence length
    """

    # Process sentence with spaCy → creates tokens + dependency tree
    doc = nlp(sentence)

    dep_lengths = []

    # Iterate over each token in the sentence
    for token in doc:

        # If token is ROOT (its head is itself), skip it
        if token.head.i == token.i:
            continue

        # Compute dependency distance:
        # token.i = position of word
        # token.head.i = position of its head
        distance = abs(token.i - token.head.i)

        dep_lengths.append(distance)

    # Compute average dependency length
    if len(dep_lengths) > 0:
        avg_dep_length = sum(dep_lengths) / len(dep_lengths)
    else:
        avg_dep_length = 0  # edge case (empty or weird sentence)

    # Return clean structured output (same format as UD parser)
    return {
        "sentence": sentence,
        "avg_dep_length": avg_dep_length,
        "sentence_length": len(doc)  # number of tokens
    }


def parse_llm(sentences):
    """
    Takes a list of sentences (LLM-generated or any raw text)
    and returns processed dependency metrics for each sentence
    """

    results = []

    # Process each sentence one by one
    for sent in sentences:
        result = process_raw_sentence(sent)
        results.append(result)

    return results