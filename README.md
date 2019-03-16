# Bag-of-Words-Meets-Bags-of-Popcorn
sentiment analysis based on word2vec and keras

### 说明
* 这是一个kaggle竞赛，数据下载的地址为 https://www.kaggle.com/c/word2vec-nlp-tutorial
* 词向量模型可以通过train_word_vectors.py文件生成
* 合并了testData,unlabeledTrainData,labeledTrainData一共100000条评论通过gensim中的CBOW生成词向量
* 使用了lstm建立神经网络，跑了20个epoch

### 训练结果
![result](https://github.com/Janeeeeeee1/Bag-of-Words-Meets-Bags-of-Popcorn/blob/master/picture1.png)

### 思考
* 可能模型并没有被fully-trained,跑更多的epoch也许会有所帮助。
* 词向量是用给定的数据集生成的，但是总共的数据量也不到100M，改进的方法可以寻找更多的电影评论的数据集，训练词向量。还可以尝试使用skip-gram和glove模型生成词向量。同时也可以尝试用其他人已经Pre-trained更好的词向量来尝试。
* 可以尝试用BiLSTM，或者更高级的attention模型。
* 可以调节lstm中神经元的个数，增加模型的深度，调节dropout_rate等参数提高准确率。
