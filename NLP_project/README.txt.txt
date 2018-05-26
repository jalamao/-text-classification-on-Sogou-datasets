运行环境：ubuntu16.04 毛廷运  2017/7/15

脚本介绍：
1. document_segmentation.py ：对SogouC文件夹下的搜狗语料库进行批量分词, 分词后的文档存放在SogouCseg文件夹下
2. feature_selection.py ：特征选择，使用了两种方：基于卡方统计和信息增益，然后计算TFIDF作为特征权重值，并将文档转换成相应格式的向量存放到文件中。
3. naivebayers_classifier.py： 训练贝叶斯分类器并且用测试语料进行测试。
4. randomforest_classifier.py：随机森林分类器训练及其测试
5. classifier_evaluate.py ：各分类器性性能评估，对各类别使用的是精确率、召回率、F1测度，对整体使用的是宏精确率、宏召回率、宏F1测度

一些重要文件：
随机森林：RF 朴素贝叶斯：NB

Feature.txt：CHI选择的特征
IG_Feature.txt：IG选择的特征

train.svm：基于CHI的libsvm的训练集
test.svm：基于CHI的libsvm的测试集

train.txt：基于CHI的随机森林的训练集
test.txt：基于CHI的随机森林的测试集

IG_train.svm：基于IG的libsvm的训练集
IG_test.svm：基于IG的libsvm的测试集

IG_train.txt：基于IG的随机森林的训练集
IG_test.txt：基于IG的随机森林的测试集

NB_predict.txt：基于CHI的NB的测试结果
RF_predict.txt：基于CHI的RF的测试结果
testscale.result:基于CHI的libsvm的测试结果

IG_testscale.result:基于IG的libsvm的测试结果
IG_NB_predict.txt:基于IG的NB的测试结果
IG_RF_predict.txt:基于IG的RF的测试结果


使用libsvm训练和预测
 使用步骤：
    1. 安装libsvm-tools
    2. 接下来可以开始训练了：
 基于CHI提取特征的SVM，命令如下：

            A. 对train.svm文件数据进行缩放到[0,1]区间

               svm-scale -l 0 -u 1 train.svm > trainscale.svm

            B. 对test.svm文件数据进行缩放到[0,1]区间

               svm-scale -l 0 -u 1 test.svm > testscale.svm

            C. 对trainscale.svm 文件进行模型训练

               svm-train -s 1 trainscale.svm trainscale.model

            D. 对testscale.svm 文件进行模型预测，得到预测结果，控制台会输出正确率

               svm-predict testscale.svm trainscale.model testscale.result

            实验结果：
                 (1)每一类训练文档：200  每一类测试数据：20   每一类特征维度：1000
                     交叉验证： python libsvm-3.22/tools/grid.py trainscale.svm ---->(best c:32, g:0.0078125)
                     svm-train -c 32 -g 0.0078125 trainscale.svm    其中：-c是惩罚因子, -g是RBF的gramma值
                     svm-predict testscale.svm trainscale.svm.model testscale.svm.predict (81.5%)

                 (2)每一类训练文档：250  每一类测试数据：每一类的可以自己计算, 总共466个   每一类特征维度：500
                    Accuracy = 84.7639% (395/466) (classification)  (换了停用词表)


 基于IG提取特征的SVM，命令如下：

            A. 对train.svm文件数据进行缩放到[0,1]区间

               svm-scale -l 0 -u 1 IG_train.svm > IG_trainscale.svm

            B. 对test.svm文件数据进行缩放到[0,1]区间

               svm-scale -l 0 -u 1 IG_test.svm > IG_testscale.svm

            C. 对trainscale.svm 文件进行模型训练

               svm-train -s 1 IG_trainscale.svm IG_trainscale.model

            D. 对testscale.svm 文件进行模型预测，得到预测结果，控制台会输出正确率

               svm-predict IG_testscale.svm IG_trainscale.model IG_testscale.result

            实验结果：
            result:  Accuracy = 84.9785% (396/466) (classification) (换了停用词表)