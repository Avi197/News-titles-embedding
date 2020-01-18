from fse import IndexedList
from fse.models.average import FAST_VERSION, MAX_WORDS_IN_BATCH
from fse.models import SIF
from gensim.models import FastText
import logging
logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)

lookup = FastText.load_fasttext_format('cc.vi.300.bin', encoding='utf-8')

sentences = []
s = IndexedList(sentences)
print(len(s))

with open('tokenized_titles_cleaned', 'r', encoding='utf-8') as file:
    for line in file:
        sentences.append(line.split())

s = IndexedList(sentences)

model = SIF(lookup, workers=2)
model.train(s)

model.save('sent2vec')
