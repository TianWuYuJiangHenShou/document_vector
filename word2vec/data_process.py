#_*_coding:utf-8_*_

from gensim.corpora import WikiCorpus
import jieba
from langconv import *
import codecs
from tqdm import tqdm,trange
import time,datetime

start = datetime.datetime.now()
zhwiki = '/NLP/data/zhwiki-latest-pages-articles.xml.bz2'
strs = []
i = 0
f = codecs.open('./zhiwiki.txt','a','utf-8')
wiki = WikiCorpus(zhwiki,lemmatize=False,dictionary={})
for text in tqdm(wiki.get_texts()):
    for sen in text:
        sen = Converter('zh-hans').convert(sen)
        sen_list = list(jieba.cut(sen))
        for s in sen_list:
            strs.append(str(s))
    tmp = ' '.join(strs)
    f.write(tmp+'\n')
    strs = []
    i = i + 1

    if(i % 200 == 0):
        print('save'+str(i)+'article')
f.close()
end = datetime.datetime.now()
print((end - start).seconds)


