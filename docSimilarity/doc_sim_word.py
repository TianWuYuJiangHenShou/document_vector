#_*_coding:utf-8_*_

from jieba import analyse,posseg
from gensim import models
import codecs
import numpy as np
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

wordvec_size = 192
'''
提取关键词
'''
def keyword_extract(data):
    keywords = analyse.extract_tags(data)
    return keywords

def getKeywords(input,output):
    outf = codecs.open(output,'a','utf8')
    input = codecs.open(input,'r','utf8')
    for data in input.readlines():
        data = data[:len(data)-1]
        keywords = keyword_extract(data)
        for word in keywords:
            outf.write(word + ' ')
        outf.write('\n')


'''
从待处理的文本中抽取关键词，并从训练好的model中获取这个词的词向量
'''

#因为提取的关键词以空格作为分割符，所以可以通过空格符的坐标来截取中文词
def get_char_pos(strs,char):
    chPos = []
    dPos = []
    try:
        chPos = list(((pos) for pos, val in enumerate(strs) if (val == char)))
        dPos = list(((val) for pos, val in enumerate(strs) if (val == char)))
    except:
        pass
    return chPos,dPos


def word2vec(file_name,model):
    file = codecs.open(file_name,'r','utf8')
    word_vec_all = np.zeros(wordvec_size)
    for data in file:
        space_pos,dpos = get_char_pos(data, ' ')
        first_word = data[0:space_pos[0]]
        '''
        实验的语料库不够大，无法包括所有的汉语词语，所以在后去一个词语的词向量时，需要判断模型中是否包含该词语
        '''
        if model.__contains__(first_word):
            word_vec_all = word_vec_all + model[first_word]
        '''
        把各行里所有词的词向量相加
        '''
        for i in range(len(space_pos) - 1):
            word = data[space_pos[i]:space_pos[i+1]]
            if model.__contains__(word):
                word_vec_all = word_vec_all + model[word]
    return word_vec_all

def simlarityCalu(vector1,vector2):
    vector1Mod=np.sqrt(vector1.dot(vector1))
    vector2Mod=np.sqrt(vector2.dot(vector2))
    if vector2Mod!=0 and vector1Mod!=0:
        simlarity=(vector1.dot(vector2))/(vector1Mod*vector2Mod)
    else:
        simlarity=0
    return simlarity

model = models.Word2Vec.load('../data/wiki_news.word2vec')
p1 = './data/P1.txt'
p2 = './data/P2.txt'
p1_keywords = './data/p1_keyword.txt'
p2_keywords = './data/p2_keyword.txt'

getKeywords(p1,p1_keywords)
getKeywords(p2,p2_keywords)
p1_vec = word2vec(p1_keywords,model)
p2_vec = word2vec(p2_keywords,model)
sim = simlarityCalu(p1_vec,p2_vec)
print(sim)