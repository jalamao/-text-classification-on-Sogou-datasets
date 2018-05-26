# coding:utf-8
__author__="maotingyun"
import numpy as np
#
# # with open("test.txt", "r") as f:
# #     f = f.readlines()
# #     for i in f:
# #         print(i)
# str = "someone@gmail.com;bill.gates@microsoft.com"
# # result = re.findall(r"[abc]", str)
# # result = re.match(r"(0[0-9]|1[0-9]|2[0-9]|[0-9])\:(0[0-9]|1[0-9]|2[0-9]|3[0-9]|4[0-9]|5[0-9]|[0-9])\:(0[0-9]|1[0-9]|2[0-9]|3[0-9]|4[0-9]|5[0-9]|[0-9])", str)
# re_email = re.compile(r"[0-9a-zA-Z]+\@[0-9a-zA-Z]+\.[0-9a-zA-Z]+")
# result = re.findall(re_email, str)
# print(result)
# print(len(result))
# with open("test.txt", "rb") as f:
#     word = open("word.txt", "rb")
#     word = word.read()
#     for line in f.readlines():
#
#
#        if line in word:
#            result = re.match(r"(.*)(\s*)(n)", line)
#            print(result)
#            result = result.group(1)
#            print(result)
#            if result:
#                print("ok")
#                F = open("test_append.txt", "ab")
#                F.write(result+"\n")
#                F.close()
#            else:
#                print("not match")
#        else:
#            print("not in")

###################################
# f = open("test_append.txt", "r")
# dir = {}
# re_str = re.compile(r"(.*)(\n)")
# for i in f:
#     i.strip()
#     i = re.match(re_str, i)
#     i = i.group(1)
#     if i in dir:
#        dir[i]+=1
#     else:
#        dir[i]=1
# print(dir)
# dir = sorted(dir.items(), key=lambda d: d[1], reverse=True)
# print(dir)
# f.close()
# with open("new.txt", "a") as f:
#     for (a,b) in dir:
#         print(a)
#         print(b)
#         f.write(a+' '+str(b)+'\n')

# ####################
#
# # # print("mao\ttingyun\tmmm")
# dir = {"毛廷运":3, "王俊俊":7}
# # for (a,b) in dir:
# #     print(a)
# #     print(b)
# a = dict()
# a = set()
# R = re.compile(r"(\t*)(\n)")
# re.split(R,)
# with open("shige.txt", "r") as f:
#    content = f.read()
#    content = content.split(" ")
#    LEN = len(content)
#    print(LEN)
#    print(content)
#    s = set()
#    for i in range(LEN):
#        s.add(content[i])
#        print(content[i])
# s1 = set("mao")
# s2 = set("wang")
# s3 = set("jun")
# # l = ["mao", "ting", "yun"]
# # ll = ["mao", "ting", "yuan"]
# # for i in l:
# #     s1.add(i)
# # print(s1)
# # for i in ll:
# #     s2.add(i)
# # print(s2)
# count = 0
# with open("new.txt", "a") as f:
#     for i in (s1|s2|s3):
#         count+=1
#         print(str(count)+" "+i)
#         f.write(str(count)+" "+i+"\n")
###########################################
# d = dict()
# print(d)
# l =["maoty", "wangjj", "yangxl"]
# count = 0
# for i in l:
#     count+=1
#     d[i] = count
# d = sorted(d.items(), key=lambda d: d[1], reverse=True)
# print(d)
###########################################
# str = "      wang    jun              jun  mao  ting  yun   "
#
# # str = str.lstrip()
# print(str)
# str = str.split()
# print(str)
# for str in str:
#    str = str.strip()
#    print(str)
#######################
# result = pow(2,4)
# l = ["mao", "zhao", "wang", "qian", "sun", "li", "mao", "mao"]
# s = set()
# for i in l:
#    s.add(i)
# print(s)
# print(s.__len__())
# for i in s:
#     print(i)
# if __name__ == "__main__":
#     print("maotingyumj")
####################################
# with open("test_append.txt", "r") as f:
#
#     f = f.read()
#     f = f.split("\n")
#     print(f)
#     print (f.__len__())
#     for i in f:
#         print(i)
#         F = open("new.txt", "a")
#         F.write(i+"\n")
############################
# l = (1, "maotingyun", "p", "90")
# i = "p"
# if i in l:
#     print("yes")
# a = tuple()
# import math
# i = math.log(100, 10)
#
# print(i)
# l = [[1, 2, 3], [3, 4, 5], [8, 9, 0]]
# result = l[:2]
# print(result)
# result = l[2:]
# print(result)
#python easy.py train.svm test.svm
# from svmutil import *
#libsvm-tools   3.12-1.1     amd64        LIBSVM binary tools
# import svm

################################################################
import jieba
import io
import os
# import sys

FolderList = ['C000007', 'C000008', 'C000010', 'C000013', 'C000014', 'C000016', 'C000020', 'C000022','C000023', 'C000024']
FolderTextCount = 300   # 在这里只选择搜狗预料库中的每个分类下的300个文本进行分词处理

ReadFilePathPrefix = os.path.dirname(os.path.abspath('__file__'))+"/SogouC/ClassFile/"  # 原始语料路径

WriteFilePathPrefix = os.path.dirname(os.path.abspath('__file__')) + "/SogouCSeg/"   # 语料被分词后的保存路径

def textcut(textname, cuttextname):
    """
    :param textname: 原始文档
    :param cuttextname: 被分词的文档
    :return: 获得分词的文档
    """
    testFile = io.open(textname, mode = 'r')
    testFileContent = testFile.readlines()
    resultfile = io.open(cuttextname, mode='w') # 避免同名文件的存在
    resultfile.close()
    resultfile = io.open(cuttextname, mode='a')
    for eachLine in testFileContent:
        eachLine = eachLine.strip('\n') # 去掉换行符“\n”
        if len(eachLine) > 0:
            #  cutResult = jieba.cut("中国的首都是北京，我在中国科学院大学。", cut_all=False)
            cutResult = jieba.cut(unicode(eachLine).encode("utf-8"), cut_all=False) # 结巴分词选择“精确模式”
            for eachword in cutResult:
                resultfile.write(eachword + " ")
            resultfile.write(u"\n")
            print(cutResult)
    resultfile.close()


# 对每个类别下的文档进行分词处理
def cut_texts_eachclass():
    print("..............Segmentation begins....................")
    for eachFolder in FolderList:
        file_count=0
        for i in range(FolderTextCount):
            # file_count
            readfilename = ReadFilePathPrefix+eachFolder+"/"+str(i)+".txt"  # 输入文档的路径
            writefilename = WriteFilePathPrefix+eachFolder+"/"+str(file_count)+".seg" #分词处理后的文档路径
            try:
                testFile = io.open(readfilename, mode='r')
                testFileContent = testFile.readlines()
                resultfile = io.open(writefilename, mode='w')  # 避免同名文件的存在
                resultfile.close()
                resultfile = io.open(writefilename, mode='a')
                for eachLine in testFileContent:
                    eachLine = eachLine.strip('\n')  # 去掉换行符“\n”
                    if len(eachLine) > 0:
                        #  cutResult = jieba.cut("中国的首都是北京，我在中国科学院大学。", cut_all=False)
                        cutResult = jieba.cut(eachLine, cut_all=False)  # 结巴分词选择“精确模式”
                        for eachword in cutResult:
                            resultfile.write(eachword + " ")
                        resultfile.write(u"\n")
                        print(cutResult)


            except Exception as err:
                print(err)
                file_count -= 1
            finally:
                file_count +=1
                resultfile.close()



    print("......................OK....................")

################################################################################
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

#信息增益计算公式
# def IGCalc(a, b, c, d):
#   C = len(ClassCode)
#   number_eachclass = float(Train_DocumentCount_Eachclass)
#   Nall = float(C * number_eachclass)
#   P = number_eachclass/Nall
#   H_C = -(C * P * math.log(P))
#   H_C_t = (float(a + b)/Nall) * () + (float(c + d)/Nall) * ()
#   IG =
#   return
import math
# 用信息增益选取特征
def featureSelection_IG(termDic, termClassDic, K):
    """    
    :param termDic: {类别1：set（[类别1的所有文档的分词], .......,类别10：set（[类别10的所有文档的分词]）}
    :param termClassDic: {类别1：[set([该类中文档0的所有分词]), ......, set([该类中文档n的所有分词])],........}
    :param K: 为每个类别选取的特征个数，比如K=500,, 那么对于10类来说,一共有10*500=5000个特征词, 经过去重后<=5000
    :return: termCountDic：每个类别下的K个特征词及其chi值构成的字典, 形如{类别1：{w1: chi, w2: chi2, ......, wk:chik},......}
    """
    C = len(ClassCode)
    number_eachclass = float(Train_DocumentCount_Eachclass)
    Nall = float(C * number_eachclass)
    all_class_set = set()
    classTermCountDic = dict()

    for key in termDic:
        classWordSets = termDic[key]
        for eachword in classWordSets:  # 对某个类别下的每一个单词的 a b c d 进行计算
            all_class_set.add(eachword)
    for eachword in all_class_set:
        H_C_t = 0
        H_C_not_t = 0
     # for key in termDic:
     #    class_local = 0
        for key in ClassCode:
            for eachclass in termClassDic:
                # key = ClassCode[class_local]
                # class_local +=1
                if eachclass == key:  # 在这个类别下进行处理
                    for eachdocset in termClassDic[eachclass]:
                        if eachword in eachdocset:
                            a = a + 1
                        else:
                            c = c + 1
                else:  # 不在这个类别下进行处理
                    for eachdocset in termClassDic[eachclass]:
                        if eachword in eachdocset:
                            b = b + 1
                        else:
                            d = d + 1
            P_C_t = float(a+1)/float(a+b+C)
            P_C_not_t = float(c+1)/float(c+d+C)
            H_C_t += P_C_t*math.log(P_C_t)
            H_C_not_t += P_C_not_t * math.log(P_C_not_t)
        P_t = float(a + b)/Nall
        P_not_t = float(c + d)/Nall
        eachwordcount_IG = math.log(float(C)) + (P_t * H_C_t + P_not_t * H_C_not_t)
        classTermCountDic[eachword] = eachwordcount_IG
    # 这个排序后返回的是元组的列表
    sortedClassTermCountDic = sorted(classTermCountDic.items(), key=lambda d: d[1], reverse=True)
    # count = 0
    termCountDic = dict()
    for i in range(K):
        termCountDic[sortedClassTermCountDic[i][0]] = sortedClassTermCountDic[i][1]

    return termCountDic  # termCountDic={{w1: IG1, w2: IG2, ......, wk:IGk}}
def writeFeatureToFile_IG(termCountDic, featurefileName):
    """    
    :param termCountDic: 每个类别下的K个特征词及其chi值构成的字典, 形如termCountDic={w1: IG1, w2: IG2, ......, wk:IGk}
    :param featurefileName: 保留特征词的文件名

    """
    featureSet = set()
    for key in termCountDic:
        featureSet.add(key)  # *******注意：集合中的元素是唯一的，所以使用集合有去除重复词的作用*********
    # featureSet=set([w1, w2, w3, ......,wn]) 其中wn是选出来的特征词，可能来自所有类
    count = 1
    file = open(featurefileName, 'w')
    for feature in featureSet:
        # 判断feature 不为空
        # stripfeature = feature.strip(" ")
        stripfeature = feature.strip()
        if len(stripfeature) > 0 and stripfeature != " ":
            file.write(str(count) + " " + stripfeature + "\n")
            count = count + 1
            print(stripfeature)
    file.close()

def stop_word(s):
    with open("stopword.txt", 'r') as f:
        all_stopword = f.readlines()
        # print(all_stopword)
        for word in all_stopword:
            word = word.strip("\n")
            if s == word:
                return True
        return False

def evaluate(test_label, predict_label, classnumber=10):
    """
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

    '''
    cut_texts_eachclass()
    '''
    # Total_Documents = 300  # 10类的总文件数量
    #
    # import os
    # for eachclass in range(len(FolderList)):
    #
    #     ReadFilePathPrefix = os.path.dirname(os.path.abspath('__file__')) + "/SogouCSeg/"
    #     path = ReadFilePathPrefix+FolderList[eachclass]
    #     count = 0
    #     for i in range(Total_Documents):
    #         f = path + "/" + str(i) + ".seg"
    #         if os.path.exists(f):
    #             count +=1
    #         else:
    #             print("don't exit " + str(i) + ".seg")
    #     print(count)
    # l = [[1, 2, 3],[4, 5,9]]
    # a = []
    # for i in l:
    #
    #     max_i = max(i)
    #     min_i = min(i)
    #     i = (i - min_i) / float((max_i - min_i))
    #     print(i)
    # l = l-[[2]]
    # print(a)
    # print(len(FolderList))
############################
# import os
# import jieba
#
# ReadFilePathPrefix = os.path.dirname(os.path.abspath('__file__'))+"/SogouC/ClassFile/C000008/111.txt"  # 原始语料路径
#
#
# try:
#     f = open(ReadFilePathPrefix, "r")
#     r = f.readlines()
#     ff = open("cut_test.txt", "w")
#     ff.close()
#     ff = open("cut_test.txt", "a")
#     print(r)
#     for eachLine in r:
#       if len(eachLine) > 0:
#            cutResult = jieba.cut(eachLine, cut_all=False)  # 结巴分词选择“精确模式”
#            # print("/".join(cutResult))
#            for eachword in cutResult:
#                print(eachword)
#                ff.write(eachword + " ")
#            ff.write("\n")
# except Exception as err:
#     print(err)
#     print("mmmmmmmmmmmmmmmmmmmmmm")
# finally:
#     print("Goodbye!")
#############################################
    # import tensorflow as tf
    # # import  numpy as np
    # sess = tf.Session()
    # input = tf.placeholder("float")
    # output = tf.nn.softmax(input)
    # out = sess.run(output, feed_dict={input:[0, 0, 0.9]})
    # print(out)
    # sess.close()
    # l1 = [[1, 3, 4, 5], [4, 5, 7, 8], [45, 5, 7, 83], [12, 3, 53, 6]]
    # l2 = [[21, 43, 4, 5],  [12, 3, 35, 63]]
    # l1, l2 = normalization(l1, l2)
    # print(l1)
    # print(l2)
    # import jieba
    #
    # eachLine="去年11月，迪志公司发现易趣网未经其许可，允许并配合其用户在网上公开拍卖该电子出版物，且这17张光盘均属盗版。"
    # cutResult = jieba.cut(eachLine, cut_all=False)
    # print("/".join(cutResult))
    # if stop_word("“"):
    #     print("wjj love you")

    # if stop_word("和"):
    #     print("wangjunjun love you")
    test_label = "test_label.txt"
    predict_label = "testscale.result"

    each_eval, macro_eval = evaluate(test_label, predict_label)
    print(each_eval)
    print(macro_eval)



