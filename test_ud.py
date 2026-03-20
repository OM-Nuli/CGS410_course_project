from src.parser.ud_parser import parse_ud

data = parse_ud("data/UD_English-EWT/en_ewt-ud-train.conllu")
print(data[0])