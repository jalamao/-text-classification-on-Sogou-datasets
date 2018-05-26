# coding:utf-8
__author__ = "maotingyun"

# import numpy as np
from sklearn import metrics
import csv
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier


Train_File_name = "IG_train.txt"  # 训练语料
Test_File_name = "IG_test.txt"  # 测试语料
# print(Train_File_name)

#随机森林参数设置
NUM_TREES = 1300   # the number of trees
MAX_FEATURES = 2  # feature


#将数据文件转换成一般的向量表示形式
def file_to_set_and_label(train_file_name, test_file_name):
    """    
    :param libsvm_train_file_name: 训练语料文件名
    :param libsvm_test_file_name: 测试语料文件名
    :return: list: set_train, label_train, set_test, label_test
    """
    train_file = open(train_file_name, "r")
    test_file = open(test_file_name, "r")
    train_lines = train_file.readlines()
    test_lines = test_file.readlines()
    train_file.close()
    test_file.close()

    label_train = list()
    label_test = list()
    set_train = list()
    set_test = list()
    ############## training set ##################

    for train_line in train_lines:
        temp = []
        # train_line = train_line.strip("\n")
        train_line = train_line.split(" ")
        # 接下来要进行格式转换 label:char-->int  set:char-->float
        label_train.append(int(train_line[0]))
        for i in train_line[1:-1]:
            i = float(i)
            temp.append(i)
        set_train.append(temp)
    # print(label_train)
    # print(set_train)
    ############### test set ###################

    for test_line in test_lines:
        temp = []
        # train_line = train_line.strip("\n")
        test_line = test_line.split(" ")
        # 接下来要进行格式转换 label:char-->int  set:char-->float
        label_test.append(int(test_line[0]))
        for i in test_line[1:-1]:
            i = float(i)
            temp.append(i)
        set_test.append(temp)
    # print(label_test)
    # print(set_test)
    return set_train, label_train, set_test, label_test

# 用于csv格式文件的保存
def write_csv(fname, jum, indexn, namelist):
    for j in range(0, jum):
        with open(fname+'.csv', 'ab') as csvfile:
            ciswriter = csv.writer(csvfile, delimiter=':', quoting=csv.QUOTE_MINIMAL)
            ciswriter.writerow([indexn[j], namelist[j]])

INDEXS = []
SCORES = []

#数据归一化处理,将数据压缩在(0, 1)之间
def normalization(set_train, set_test):
    train_temp = []
    test_temp = []
    #训练数据归一化
    for i in set_train:
        i = np.array(i)
        max_i = max(i)
        min_i = min(i)
        i = (i - min_i)/(max_i - min_i)
        train_temp.append(list(i))
    #测试数据归一化
    for i in set_test:
        i = np.array(i)
        max_i = max(i)
        min_i = min(i)
        i = (i - min_i)/(max_i - min_i)
        test_temp.append(list(i))
    return train_temp, test_temp


if __name__ == "__main__":
    # 获取训练数据集、训练数据标签, 测试数据集、测试数据标签
    print("............obtain data and label............")
    set_train, label_train, set_test, label_test = file_to_set_and_label(Train_File_name, Test_File_name)
    print("..............data normalization................")
    set_train, set_test = normalization(set_train, set_test)
    # Random Forest
    print(".............training............")
    clf = RandomForestClassifier(n_estimators=NUM_TREES, max_features=MAX_FEATURES) #构建随机森林分类器
    clf.fit(set_train, label_train)  # 加载训练数据和标签, 开始训练

    # print(".............save model............")
    # with open("random_forest.pkl", "wb") as f:
    #     pickle.dump(clf, f)
    #
    # print(".......use model for predicting............")
    # with open("random_forest.pkl", "rb") as f:
    #     clf = pickle.load(f)

    print(".............predicting..............")
    predict = clf.predict(set_test)
    #
    # with open("IG_RF_predict.txt", "a") as f:
    #     for i in predict:
    #         f.write(str(i) + "\n")

    accuracy = metrics.accuracy_score(label_test, predict)
    print("accuracy: ", accuracy)

    print(".............storing accuracy............")
    SCORES.append(accuracy)
    str_rf = "the number of trees:" + str(NUM_TREES)+ "  max_feature:" + str(MAX_FEATURES)+ "  accuracy"
    INDEXS.append(str_rf)
    write_csv("RF_accuracy", len(SCORES), INDEXS, SCORES)  # records accuracy


    # kfold = model_selection.KFold(n_splits=10, random_state=SEED)
    # results = model_selection.cross_val_score(clf, output_VGG_test, Y_RF_test, cv=kfold)  # Use cross validation
    # acc = results.mean()
    # print("results", results)
    # print('RF accuracy: ', acc)
    # with open("test_label.txt", 'a') as f:
    #     for label in label_test:
    #         f.write(str(label) + '\n')