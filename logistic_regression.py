# coding=utf-8
from __future__ import print_function
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from scipy.sparse import hstack

warnings.filterwarnings('ignore')

data_path = r'data/train.tsv'


def get_train_val_data(data_path):
    '''
    使用one-hot的bag of word和tf-idf的bag of word来对文本进行数字化
    :param data_path:
    :return:
    '''

    # 读取训练集，并显示样本
    train_raw_data = pd.read_csv(data_path, sep='\t')

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(train_raw_data['Phrase'],
                                                        train_raw_data['Sentiment'],
                                                        test_size=0.2,
                                                        random_state=6)
    # 训练集、测试集数目
    print('train data size:', len(X_train))
    print('test data size:', len(X_test))

    # 采用2-gram将文本向量化
    vectorizer = CountVectorizer(max_features=10000,  # 词汇表数目
                                 ngram_range=(1, 2),  # 采用1-gram和2-gram
                                 stop_words='english')  # 使用内置停用词

    # bag of word文本数字化
    X_train_count = vectorizer.fit_transform(X_train)
    X_test_count = vectorizer.transform(X_test)

    # tf-idf文本数字化
    tfidf_ngram_vec = TfidfVectorizer(max_features=10000,
                                      ngram_range=(1, 2),
                                      stop_words='english')

    X_train_tfidf_ngram = tfidf_ngram_vec.fit_transform(X_train)
    X_test_tfidf_ngram = tfidf_ngram_vec.fit_transform(X_test)

    # 合并特征可以提升效果
    X_train_tfidf = hstack([X_train_count, X_train_tfidf_ngram])
    X_test_tfidf = hstack([X_test_count, X_test_tfidf_ngram])
    print('X_train_tfidf shape:', X_train_tfidf.shape, 'X_test_tfidf shape:', X_test_tfidf.shape)

    return X_train_tfidf, y_train, X_test_tfidf, y_test


def train_eval():
    X_train_tfidf, y_train, X_test_tfidf, y_test = get_train_val_data(data_path)

    # 构建多项式Logistic regression
    clf = LogisticRegression(solver='sag',
                             max_iter=100,
                             random_state=88,
                             multi_class='ovr',
                             verbose=1)

    # 模型训练
    clf.fit(X_train_tfidf, y_train)

    # 训练集上效果
    print("training score : {0:.3f} (multinomial)".format(clf.score(X_train_tfidf, y_train)))

    # 模型评估
    y_pred = clf.predict(X_test_tfidf)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))


if __name__ == '__main__':
    train_eval()
