import pandas as pd
import re
from bs4 import BeautifulSoup
from string import punctuation
from gensim.models import word2vec

unlabeledTrainData = pd.read_csv('./word2vec-nlp-tutorial/unlabeledTrainData.tsv',sep='\t',escapechar='\\')
labeledTrainData = pd.read_csv('./word2vec-nlp-tutorial/labeledTrainData.tsv',sep='\t',escapechar='\\')
testData =  pd.read_csv('./word2vec-nlp-tutorial/testData.tsv',sep='\t',escapechar='\\')
labeledTrainData = labeledTrainData.drop(['sentiment','id'],axis=1)
unlabeledTrainData = unlabeledTrainData.drop(['id'],axis=1)
testData = testData.drop(['id'],axis=1)
corpus = pd.concat([labeledTrainData,unlabeledTrainData,testData],axis=0,ignore_index=True)

stopwords = []
with open('./stopwords.txt','r',encoding='utf-8') as f:
    for line in f.readlines():
        line = line.strip()
        stopwords.append(line)

# 数据清理
punc = punctuation + u'.,;《》？！“”‘’@#￥%…&×（）——+【】{};；●，。&～、|\s:：'
def clean_review(review):
    review = BeautifulSoup(review,'html.parser').get_text()
    review = re.sub(r"[{}]+".format(punc)," ",review)
    review = review.lower().split()
    words = [w for w in review if w not in stopwords]
    return ' '.join(words)

corpus['review_cleaned'] = corpus['review'].apply(clean_review)

corpus_txt = open('corpus_txt.txt','w',encoding='utf-8')
for i in range(len(corpus)):
    corpus_txt.write(str(corpus['review_cleaned'][i])+'\n')
corpus_txt.close()

sentences=word2vec.LineSentence(u'corpus_txt.txt')
model=word2vec.Word2Vec(sentences,min_count=1,size=100,negative=10)
model.save("review_word_vectors.model")