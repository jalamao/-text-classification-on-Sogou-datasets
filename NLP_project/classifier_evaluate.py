#coding:utf-8
__author__ = "maotingyun"
import numpy as np
test_label_file = "test_label.txt" # 测试集的标签
NB_test_label_file = "NB_test_label.txt"
SVM_predict_label_file = "testscale.result" # SVM的预测结果
RF_predict_label_file = "RF_predict.txt"#"RF_predict.txt" # 随机森林的预测结果
NB_predict_label_file = "NB_predict.txt" # 朴素贝叶斯的预测结果

def evaluate(test_label, predict_label, classnumber=10):
    """
    用于评价分类器的性能, 单个类别的评价指标是精确度, 找回率, F1测度
    整个类别的评价指标是宏精确度, 宏找回率, 宏F1测度
    :param test_label: 存放测试文本类别标签的文件
    :param predict_label: 存放分类器输出的类别的文件
    :param classnumber: 类别数
    :return: 每个类别的precision, recall, F1 和宏P, 宏R, 宏F1
    """
    real_label = open(test_label, "r")
    classifier_label = open(predict_label, "r")
    test_label_list = list()
    predict_label_list = list()
    for label in real_label:
        label = label.strip("\n")
        label = int(label)
        test_label_list.append(label)
    for label in classifier_label:
        label = label.strip("\n")
        label = int(label)
        predict_label_list.append(label)
    eachclass_evaluate = list()
    P_macro = 0
    R_macro = 0
    for i in range(classnumber):
        start_index = test_label_list.index(i)
        offset = test_label_list.count(i)
        result = np.equal(test_label_list[start_index:start_index + offset], predict_label_list[start_index:start_index + offset])
        result = list(result)
        a = result.count(True)
        c = offset - a
        b = predict_label_list.count(i) - a
        P = float(a) / float(a + b) #precision
        R = float(a) / float(a + c) #recall
        F1 = 2 * P * R / (P + R) #F1-measure
        eachclass_evaluate.append([P, R, F1])
        P_macro += P
        R_macro += R
    P_macro = P_macro / classnumber # 取平均精确率
    R_macro = R_macro / classnumber # 取平均找回率
    F1_macro = 2 * P_macro * R_macro / (P_macro + R_macro) # 求宏F1
    return eachclass_evaluate, [P_macro, R_macro, F1_macro]

if __name__ == "__main__":

    flag = "RF"  # 可选"NB" "SVM" "RF" 选择相应标志，对相应的分类器作评估

    if flag == "NB":
        eachclass_eval, macro_eval = evaluate(NB_test_label_file, NB_predict_label_file)
        print(eachclass_eval)
        print(macro_eval)
    if flag == "SVM":
        eachclass_eval, macro_eval = evaluate(test_label_file, SVM_predict_label_file)
        print(eachclass_eval)
        print(macro_eval)
    if flag == "RF":
        eachclass_eval, macro_eval = evaluate(test_label_file, RF_predict_label_file)
        print(eachclass_eval)
        print(macro_eval)




