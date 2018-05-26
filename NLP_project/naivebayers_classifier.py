# coding:utf-8
__author__ = "maotingyun"
from feature_selection import readFileToList, readFeature, textCutBasePath, ClassCode, Train_DocumentCount_Eachclass
import math

#将字典内部的列表转换成集合
def list_to_set(train_dic, test_dic):
    train_temp_dic = dict()
    test_temp_dic = dict()
    #train
    for key in train_dic:
        train_list = list()
        train_dic_list = train_dic[key]
        for l in train_dic_list:
            temp_set = set()
            for word in l:
                temp_set.add(word)
            train_list.append(temp_set)
        train_temp_dic[key] = train_list
    # test
    for key in test_dic:
        test_list = list()
        test_dic_list = test_dic[key]
        for l in test_dic_list:
            temp_set = set()
            for word in l:
                temp_set.add(word)
            test_list.append(temp_set)
        test_temp_dic[key] = test_list
    return train_temp_dic, test_temp_dic  # train_temp_dic/test_temp_dic = {类别1：[set1, set2,...,setn],...,类别10：[set1, set2,...,setn]}

# 用朴素贝叶斯分类
def NaiveByayes_Classifier(train_temp_dic, test_temp_dic, feature):
    feature_set = set()
    for word in feature:
        feature_set.add(word)
    NB_predict = list()
    NB_test_label = list()
    correction_count = 0
    for label in test_temp_dic:
        test_list = test_temp_dic[label]

        for a_test_set in test_list:
            P_list = []
            C_list =[]
            for c in train_temp_dic:
                train_list = train_temp_dic[c]
                P_W_C = 0
                for word in a_test_set:
                    if word in feature_set:
                        a = 0
                        for train_set in train_list:
                            if word in train_set:
                                a += 1
                        P_W_C += math.log(float(a + 1) / float(Train_DocumentCount_Eachclass + len(feature))) #计算对数似然

                P_list.append(P_W_C)
                C_list.append(c)
            max_index = P_list.index(max(P_list))
            predict_label = C_list[max_index]
            nb_predict_label = ClassCode.index(predict_label)
            NB_test_label.append(ClassCode.index(label))
            NB_predict.append(nb_predict_label)
            if  predict_label == label:
                correction_count += 1
    accuracy = float(correction_count)/float(466)
    return accuracy, NB_predict, NB_test_label


if __name__ == "__main__":
    print("..................preprocessing...................")
    # train_dic/test_dic={类别1：[[本类中文档0的词],[本类中文档1的词],......,[本类中文档n的词]],......., 类别10：[[本类中文档0的词],[本类中文档1的词],......,[本类中文档n的词]]}
    train_dic, test_dic = readFileToList(textCutBasePath, ClassCode, Train_DocumentCount_Eachclass)
    feature = readFeature("IG_Feature.txt") # feature=[w1, w2, w3,......,wn], 其中wn是特征词
    train_temp_dic, test_temp_dic = list_to_set(train_dic, test_dic)
    # print(len(test_temp_dic['C000007']))
    print("..................Naive Bayes predicts accuracy...................")
    accuracy, NB_predict, NB_test_label = NaiveByayes_Classifier(train_temp_dic, test_temp_dic, feature)
    print("accuracy of NB: ", accuracy)
    # with open("IG_NB_predict.txt", "a") as f:
    #     for i in NB_predict:
    #         f.write(str(i) + "\n")
    """
    备注：CHI提取的特征,len(feature)=4824,  accuracy of NB: 0.7703862660944206
         IG提取的特征,len(feature)=4824,   accuracy of NB: 0.6866952789699571
         修改停用词表后, CHI提取的特征, len(feature)=4840 accuracy of NB: 0.8090128755364807
          修改停用词表后, IG提取的特征, len(feature)=4840 accuracy of NB: 0.723175965665236
    """