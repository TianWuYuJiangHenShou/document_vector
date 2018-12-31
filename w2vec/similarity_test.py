#_*_coding:utf-8_*_

import gensim
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

models = gensim.models.Word2Vec.load('../data/wiki_news.word2vec')
print(models.similarity('土豆','紫薯'))
print(models.similarity('红薯','紫薯'))

word = '扬州'
if word in models.wv.index2word:
    print(models.most_similar(word))