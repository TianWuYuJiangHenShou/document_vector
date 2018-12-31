#_*_coding:utf-8_*_

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import codecs
import multiprocessing
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

wiki_sen = codecs.open('../data/zhiwiki.txt','r','utf-8')
#workers  = multiprocessing.cpu_count()  设置成这个，满CPU运行程序
model = Word2Vec(LineSentence(wiki_sen),sg=0,size=192,window=5,min_count=5,workers=multiprocessing.cpu_count()-1)
model.save('./wiki_news.wocrd2vec')