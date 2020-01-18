from fse.models import base_s2v
from fse import IndexedList
model = base_s2v.BaseSentence2VecModel.load('sent2vec')
sentences = []
s = IndexedList(sentences)
print(len(s))

with open('tokenized_titles_cleaned', 'r', encoding='utf-8') as file:
    for line in file:
        sentences.append(line.split())

s = IndexedList(sentences)

print(s[0])
print(model.sv[0])
print(model.sv.most_similar(0, indexable=s.items))

