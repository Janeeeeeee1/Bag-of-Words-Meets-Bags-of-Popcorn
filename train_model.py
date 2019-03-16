from gensim.models import word2vec
import keras
from train_word_vectors import clean_review
import numpy as np
import pandas as pd
from keras.layers import Embedding, LSTM,Dropout, Dense, Input
from keras.models import Model
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

# 加载模型
wordvectors = word2vec.Word2Vec.load("review_word_vectors.model")
vocabulary = wordvectors.wv.vocab.keys()
vocabulary= list(vocabulary)
embeddings = np.load('review_word_vectors.model.wv.vectors.npy')

labeledTrainData = pd.read_csv('./word2vec-nlp-tutorial/labeledTrainData.tsv',sep='\t',escapechar='\\')
testData =  pd.read_csv('./word2vec-nlp-tutorial/testData.tsv',sep='\t',escapechar='\\')
labeledTrainData['review_cleaned']=labeledTrainData['review'].apply(clean_review)
testData['review_cleaned']=testData['review'].apply(clean_review)

#把labeledTrainData中的每一条评论取出来放到trainData_txt.txt中
trainData_txt = open('trainData_txt.txt','w',encoding='utf-8')
for i in range(len(labeledTrainData)):
    temp = labeledTrainData['review_cleaned'][i]
    trainData_txt.write(str(temp)+'\n')
trainData_txt.close()
#把labeledTrainData中的每一条评论取出来放到trainData_txt.txt中
testData_txt = open('testData_txt.txt','w',encoding='utf-8')
for i in range(len(testData)):
    temp = testData['review_cleaned'][i]
    testData_txt.write(str(temp)+'\n')
testData_txt.close()

# 把数据放到列表中[['a','b'],['a','c']...] 内部每一个列表是一句话
with open('trainData_txt.txt','r',encoding='utf-8') as f:
    lineList_train = []
    for line in f.readlines():
        wordList = line.strip().split(" ")
        lineList_train.append(wordList)
with open('testData_txt.txt','r',encoding='utf-8') as f:
    lineList_test = []
    for line in f.readlines():
        wordList = line.strip().split(" ")
        lineList_test.append(wordList)

# training data labels
labels = list(labeledTrainData['sentiment'])

# lineList_train转化成数字
lineList_train_number = np.copy(lineList_train)
for line in range(len(lineList_train)):
    for w in range(len(lineList_train[line])):
        lineList_train_number[line][w] = vocabulary.index(lineList_train[line][w])
# lineList_test转化成数字
lineList_test_number = np.copy(lineList_test)
for line in range(len(lineList_test)):
    for w in range(len(lineList_test[line])):
        lineList_test_number[line][w] = vocabulary.index(lineList_test[line][w])

# 截取前250个单词，不够的用0填充，0对应的单词为'stuff'
data_train = keras.preprocessing.sequence.pad_sequences(lineList_train_number,maxlen=250,dtype='int32',padding='post')
data_test = keras.preprocessing.sequence.pad_sequences(lineList_test_number,maxlen=250,dtype='int32',padding='post')

def Lstm_model(embeddings):
    input_layer = Input(shape=(250,), dtype='int32')
    embedding_layer = Embedding(input_dim=len(embeddings), output_dim=len(embeddings[0]),
                                weights=[embeddings],  # 表示直接使用预训练的词向量
                                trainable=False)(input_layer)  # False表示不对词向量微调
    Lstm_layer = LSTM(units=256, return_sequences=False)(embedding_layer)
    drop_layer = Dropout(0.5)(Lstm_layer)
    dense_layer = Dense(units=1, activation="sigmoid")(drop_layer)

    model = Model(inputs=[input_layer], outputs=[dense_layer])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


os.environ["CUDA_VISIBLE_DEVICES"]="1,2" # 使用编号为1，2号的GPU
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7 # 每个GPU现存上届控制在60%以内
session = tf.Session(config=config)
KTF.set_session(session)

model = Lstm_model(embeddings)
model.fit(data_train,labels,epochs=100, batch_size=64, verbose=1)
y_pred = model.predict(data_test)
y_pred_idx = [1 if prob[0] > 0.5 else 0 for prob in y_pred]

# 导出到csv文件中提交
results = pd.Series(y_pred_idx,name="sentiment")
submission = pd.concat([testData['id'],results],axis = 1)
submission.to_csv("Comment_analysis.csv",index=False)


