from src.parser.ud_parser import read_conllu, process_sentence

data = read_conllu("data/UD_English-EWT/en_ewt-ud-train.conllu")

sample = data[0]

result = process_sentence(sample["tokens"], sample["heads"])

print(result)