from fse.models import base_s2v
from fse import IndexedList
model = base_s2v.BaseSentence2VecModel.load('sent2vec')
sentences = []
s = IndexedList(sentences)

title_file = 'H:/Vietnamese word representations/News-titles-embedding/Data/tokenized_titles_cleaned'

with open(title_file, 'r', encoding='utf-8') as file:
    for line in file:
        sentences.append(line.split())

s = IndexedList(sentences)

# model.sv.save('H:/Vietnamese word representations/News-titles-embedding/sent_vec')

model.sv.similar_by_sentence("Is this really easy to learn".split(), model=model, indexable=s.items)

print(s[10])
print(model.sv[21758])
print(model.sv.most_similar(10, indexable=s.items))

