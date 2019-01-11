# 第一章：深度学习介绍

## 1.自然语言应用

在自然语言处理领域， 一个非常棘手的问题在于自然语言中有很多词表达了相近的意
思，比如“狗”和“犬” 就几乎表达了同样的意思。然而“狗”和“犬”的编码在计算机
中可能差别很大，所以计算机就无法很好地理解自然语言所表达的语义。为了解决这个问
题，研究人员人工建立了大量的语料库。通过这些语料库，可以大致刻画自然语言中单词
之间的关系。在建立好的语料库中， WordNet©、ConceptNet®和FrameNet ®是其中影响力比
较大的几个。然而语料库的建立需要花费很多人力物力，而且扩展能力有限。单词向量提
供了一种更加灵活的方式来刻画单词的语义。

单词向量会将每一个单词表示成一个相对较低维度的向量（比如100 维或200 维）。对
于语义相近的单词， 其对应的单词向量在空间中的距离也应该接近。于是单词语义上的相
似度可以通过空间中的距离来描述。单词向量不需要通过人工的方式设定，它可以从互联
网上海量非标注文本中学习得到。使用斯坦福大学开源的GloVe⑦单词向量可以得到与单词
“企og （青蛙）”所对应的单词向量最相似的5 个单词分别是“企ogs （ 青蛙复数）”、“ toad （蜡
蛤） ”、“ litoria （ 雨滨蛙属） ”、“ leptodactylidae （细趾蜡科）”和“ rana （中国林蛙） ” 。从这
个样例可以看出， 单词向盘’可以非常有效地刻画单词的语义。通过单词向量还可以进行单
词之间的运算。比如用单词“ king ”所代表的向量减去单词“ man ”所代表的向量得到的结
果向量和单词“ queen ”减去“ woman ”得到的结果向量相似。这说明在单词向量中，己经
隐含了表达性别的概念。