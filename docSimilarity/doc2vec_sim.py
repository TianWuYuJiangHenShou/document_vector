#_*_coding:utf-8_*_

from gensim import models
import codecs
import numpy as np
import jieba
import logging
#enable logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


model_path = './data/zhiwiki_news.doc2vec'
start_alpha = 0.01
infer_epoch = 1000
docvec_size = 192

def simCalu(vec1,vec2):
    vec1Mod = np.sqrt(vec1.dot(vec1))
    vec2Mod = np.sqrt(vec2.dot(vec2))
    if vec1Mod != 0 and vec2Mod != 0:
        sim = (vec1.dot(vec2)) / (vec1Mod * vec2Mod)
    else:
        sim = 0
    return sim

def doc2vec(file_name,model):
    doc = [w for x in codecs.open(file_name,'r','utf-8').readlines() for w in jieba.cut(x.strip())]
    doc_vec_all = model.infer_vector(doc,alpha=start_alpha,steps=infer_epoch)
    return doc_vec_all


model = models.Doc2Vec.load(model_path)
p1 = './data/P1.txt'
p2 = './data/P2.txt'
p1_doc2vec = doc2vec(p1,model)
p2_doc2vec = doc2vec(p2,model)
print(simCalu(p1_doc2vec,p2_doc2vec))
