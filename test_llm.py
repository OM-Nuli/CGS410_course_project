from src.parser.llm_parser import parse_llm

sentences = [
    "The dog runs",
    "A cat sleeps on the mat"
]

data = parse_llm(sentences)

for d in data:
    print(d)