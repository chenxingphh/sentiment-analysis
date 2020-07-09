# coding=utf-8
from __future__ import print_function
from tensorflow import keras
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.utils.np_utils import to_categorical

data_path = r'data/train.tsv'
n_class = 6  # 类别数目


def decode_review(text, reverse_word_index):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


def get_max_len(sentences):
    return max([len(sent) for sent in sentences])


def get_data():
    # 读取数据集
    train_raw_data = pd.read_csv(data_path, sep='\t')
    all_data = train_raw_data['Phrase']
    all_label = train_raw_data['Sentiment']

    # 将word映射为数字
    token = keras.preprocessing.text.Tokenizer(num_words=16000,
                                               lower=True,
                                               split=" ",
                                               char_level=False)
    token.fit_on_texts(all_data)
    all_encode_data = token.texts_to_sequences(all_data)

    # 获取最大句子长度
    max_len = get_max_len(all_encode_data)
    print('Max input len:', max_len)

    # 标准化填充
    all_encode_data = keras.preprocessing.sequence.pad_sequences(all_encode_data,
                                                                 value=0,
                                                                 padding='post',
                                                                 maxlen=max_len)

    # 标签one-hot处理
    all_label = to_categorical(all_label, num_classes=n_class)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(all_encode_data, all_label, test_size=0.2, random_state=66)
    print('X_train.shape:', X_train.shape)
    print('X_test.shape:', X_test.shape)

    # 获取词映射字典
    word_index = token.word_index
    index_word = dict([(value, key) for (key, value) in token.word_index.items()])

    # 填充一个未知的word
    word_index['<UNK>'] = 0
    index_word[0] = '<UNK>'
    print('Vocab size:', len(word_index.keys()))

    return X_train, X_test, y_train, y_test, word_index, index_word


if __name__ == '__main__':
    # 获取的标签是单个数字而不是one-hot类型
    X_train, X_test, y_train, y_test, word_index, index_word = get_data()

    print('Encode text\n', X_train[0])
    print('Decode text', decode_review(X_train[0], index_word))
