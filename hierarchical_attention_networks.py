# coding=utf-8
from __future__ import print_function
from tensorflow.keras.layers import Embedding, Dense, GRU, Layer, Bidirectional, Input
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras import optimizers, Sequential
import data_util as du
from tensorflow.keras import initializers

vocab_size = 15289  # 词汇量
word_dim = 50  # 词向量维度
n_class = 6  # 类别数目
max_input_len = 48
model_path = 'model/lstm_model.h5'  # 模型存储路径


class HierarchicalAttention(Layer):
    '''
    HAN论文是针对Text,即多个句子；
    由于这里输入仅仅只是单个句子，因此只能实现HAN中的word attention,仅得到单个s
    '''

    def __init__(self, att_dim):
        super(HierarchicalAttention, self).__init__()
        self.att_dim = att_dim
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) == 3

        # inputs shape:[batch,input_len,input_dim]
        self.W = Dense(self.att_dim, activation='tanh')

        self.u_w = self.add_weight(name='context_vector',
                                   shape=(self.att_dim, 1),
                                   initializer='uniform',
                                   trainable=True)

    def call(self, h_t, mask=None):
        # h_t:[batch,input_len,input_dim]
        # U:[batch,input_len,att_dim]
        U = self.W(h_t)

        # U*u_w,alpha:[batch,input_len,1]
        alpha = K.dot(U, self.u_w)
        alpha = K.squeeze(alpha, axis=-1)

        # alpha:[batch,input_len]
        alpha = K.exp(alpha)

        # mask的值为0,经过相乘后再进行softmax时注意力权重也为0
        if mask is not None:
            alpha *= K.cast(mask, tf.float32)

        # 进行sotfmax操作
        alpha /= K.cast(K.sum(alpha, axis=1, keepdims=True) + K.epsilon(), tf.float32)

        # 权重值与对应的向量进行乘
        weighted_h_t = K.dot(alpha, h_t)

        # 与输入的h_t进行相乘和相加
        # return shape[batch,input_dim]
        output = K.sum(weighted_h_t, axis=1)
        return output

    def compute_mask(self, inputs, mask=None):
        return mask


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


def han_model():
    model = Sequential()
    model.add(Embedding(vocab_size, word_dim, mask_zero=True, input_length=max_input_len))
    model.add(Bidirectional(GRU(16, activation='tanh', return_sequences=True)))
    model.add(HierarchicalAttention(50))
    model.add(Dense(n_class, activation='softmax'))

    # 配置优化器和损失函数
    rmsp = optimizers.RMSprop(lr=0.01)
    model.compile(optimizer=rmsp,
                  loss='categorical_crossentropy',
                  metrics=['accuracy', Precision, Recall])

    model.summary()
    return model


def train(X_train, X_test, y_train, y_test, epoch, model_exist):
    model = han_model()

    if model_exist:
        model.load_weights(model_path)

    history = model.fit(X_train, y_train, epochs=epoch, batch_size=128, validation_data=(X_test, y_test))
    print('best val_acc', max(history.history['val_acc']))

    # 保存模型
    model.save(model_path)


if __name__ == '__main__':
    X_train, X_test, y_train, y_test, word_index, index_word = du.get_data()
    train(X_train, X_test, y_train, y_test, epoch=10, model_exist=False)
