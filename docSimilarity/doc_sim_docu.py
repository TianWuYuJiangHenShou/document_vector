#_*_coding:utf-8_*_

from gensim import models,corpora
import jieba
import codecs
import logging
from langconv import *
#enable logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


zhwiki = '/Users/yangyang/Desktop/NLP/data/zhwiki-latest-pages-articles.xml.bz2'
wiki = corpora.WikiCorpus(zhwiki,lemmatize=False,dictionary={})

'''
gensim LabeledSentence：将文本(分词)、标签一起训练，得到文本向量
'''
class TaggedWikiDocument(object):
    def __init__(self, wiki):
        self.wiki = wiki
        self.wiki.metadata = True
    def __iter__(self):
        for content, (page_id, title) in self.wiki.get_texts():
            yield models.doc2vec.LabeledSentence(words=[w for c in content for w in jieba.cut(Converter('zh-hans').convert(c))], tags=[title])


documents = TaggedWikiDocument(wiki)
model = models.Doc2Vec(documents,dm=0,window=8,dbow_words=1,size=192,min_alpha=19,iter=5,workers=6)
model.save('./data/zhiwiki_news.doc2vec')