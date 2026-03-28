python -c "
import stanza
nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma,depparse', tokenize_pretokenized=True, verbose=False)
doc = nlp('The dog runs in the park')
for sent in doc.sentences:
    for word in sent.words:
        print(word.text, word.head)
print('SUCCESS')
"