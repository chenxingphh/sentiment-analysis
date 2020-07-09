# coding=utf-8
from __future__ import print_function
from tensorflow.keras.layers import GlobalAveragePooling1D, Embedding, Dense
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers, Sequential
import data_util as du

vocab_size = 15289  # 词汇量
word_dim = 50  # 词向量维度
n_class = 6  # 类别数目
model_path = 'model/fast_text_model.h5'  # 模型存储路径

def Precision(y_true, y_pred):
    '''
    精确率
    :param y_true: one-hot类型
    :param y_pred:
    :return:
    '''
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  # true positives
    pp = K.sum(K.round(K.clip(y_pred, 0, 1)))  # predicted positives
    precision = tp / (pp + K.epsilon())
    return precision


def Recall(y_true, y_pred):
    '''
    召回率
    :param y_true: one-hot类型
    :param y_pred:
    :return:
    '''
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  # true positives
    pp = K.sum(K.round(K.clip(y_true, 0, 1)))  # possible positives
    recall = tp / (pp + K.epsilon())
    return recall

def fast_text_model():
    model = Sequential()
    model.add(Embedding(vocab_size, word_dim))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(n_class, activation='softmax'))

    # 配置优化器和损失函数
    adam = optimizers.Adam(lr=0.01)
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy', Precision, Recall])

    model.summary()
    return model


def train(X_train, X_test, y_train, y_test, epoch, model_exist):
    model = fast_text_model()

    if model_exist:
        model.load_weights(model_path)

    history = model.fit(X_train, y_train, epochs=epoch, batch_size=256, validation_data=(X_test, y_test))

    print('fast_text_model best val_acc ', max(history.history['val_acc']))

    # 保存模型
    model.save(model_path)


if __name__ == '__main__':
    X_train, X_test, y_train, y_test, word_index, index_word = du.get_data()
    train(X_train, X_test, y_train, y_test, epoch=10, model_exist=False)
