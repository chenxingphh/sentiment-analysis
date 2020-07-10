# sentiment-analysis
采用常用的机器学习和深度学习方法来进行情感数据分类

# 数据集
烂番茄情感分析数据集：https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews。烂番茄影评数据集是一个用于情感分析的影评语料库。Kaggle比赛提供了一个在烂番茄数据集上对你的情绪分析
想法进行基准测试的机会每个句子都被斯坦福大学的语法分析器分解成许多短语。每个短语都有一个短语id。每个句子都有一个句子id。重复的短语（如短词/常用词）只包含在数据中一次。

由于没有提供测试集的标签，因此将训练集的部分数据划分为测试集

## 数据实例
    PhraseId	SentenceId	Phrase	Sentiment
    1	1	A series of escapades demonstrating the adage that what is good for the goose is also good for the gander , some of which occasionally amuses but none of which amounts to much of a story .	1
    2	1	A series of escapades demonstrating the adage that what is good for the goose	2

## 标签
    0 - negative
    1 - somewhat negative
    2 - neutral
    3 - somewhat positive
    4 - positive

# 包依赖
* Python3.6
* Tensorflow 1.9.0

