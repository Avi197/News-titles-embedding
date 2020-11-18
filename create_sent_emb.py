from fse import IndexedList
from fse.models.average import FAST_VERSION, MAX_WORDS_IN_BATCH
from fse.models import SIF
from gensim.models import FastText
import logging
logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)

w2v_model = "H:/Vietnamese word representations/Word_vector_data/VnNewsWord2Vec/VnNewsWord2Vec.bin"

lookup = FastText.load_fasttext_format(w2v_model, encoding='utf-8')

sentences = []
s = IndexedList(sentences)
print(len(s))

title_file = 'H:/Vietnamese word representations/News-titles-embedding/Data/tokenized_titles_cleaned'

with open(title_file, 'r', encoding='utf-8') as file:
    for line in file:
        sentences.append(line.split())

s = IndexedList(sentences)

model = SIF(lookup, workers=2)
model.train(s)

model.save('sent2vec')
