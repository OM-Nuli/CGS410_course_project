from src.parser.random_parser import parse_random

sentences = [
    "The dog runs",
    "A cat sleeps on the mat"
]

data = parse_random(sentences)

for d in data:
    print(d)