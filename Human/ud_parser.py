from conllu import parse_incr

# Step 1: Read .conllu file
def read_conllu(file_path):
    sentences = []

    with open(file_path, "r", encoding="utf-8") as f:
        for tokenlist in parse_incr(f):
            tokens = []
            heads = []

            for token in tokenlist:
                # Only include real tokens (skip ranges like 1-2)
                if isinstance(token["id"], int):
                    tokens.append(token["form"])
                    heads.append(token["head"])

            sentences.append({
                "tokens": tokens,
                "heads": heads
            })

    return sentences

# Step 2: Compute dependency length
def compute_dependency_lengths(tokens, heads):
    dep_lengths = []

    for i, head in enumerate(heads):
        if head == 0:
            continue  # skip ROOT

        # i is 0-based, UD heads are 1-based
        distance = abs((i + 1) - head)
        dep_lengths.append(distance)

    return dep_lengths

# Step 3: Process a single sentence
def process_sentence(tokens, heads):
    dep_lengths = compute_dependency_lengths(tokens, heads)

    if len(dep_lengths) > 0:
        avg_dep_length = sum(dep_lengths) / len(dep_lengths)
    else:
        avg_dep_length = 0

    return {
        "sentence": " ".join(tokens),
        "avg_dep_length": avg_dep_length,
        "sentence_length": len(tokens)
    }

# Step 4: Process entire UD file
def parse_ud_file(file_path):
    raw_sentences = read_conllu(file_path)

    processed_data = []

    for sent in raw_sentences:
        result = process_sentence(
            sent["tokens"],
            sent["heads"]
        )
        processed_data.append(result)

    return processed_data

def parse_ud(file_path):
    return parse_ud_file(file_path)