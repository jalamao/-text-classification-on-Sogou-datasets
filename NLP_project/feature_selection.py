# coding:utf-8
import math
import sys
import os
# import io
# 使用开方统计选择特征
# 按UTF-8编码格式读取文件

# 定义停用词
# def stop_word(s):
#     return s == ' ' or s == '/t' or s == '/n' \
#            or s == '，' or s == '。' or s == '！' or s == '、' or s == '―' \
#            or s == '？' or s == '＠' or s == '：'\
#            or s == '＃' or s == '%' or s == '＆' or s=="\’" or s=="\‘"  \
#            or s == '（' or s == '）' or s == '《' or s == '》' \
#            or s == '［' or s == '］' or s == '｛' or s == '｝' \
#            or s == '*' or s == ',' or s == '.' or s == '&' \
#            or s == '!' or s == '?' or s == ':' or s == ';' \
#            or s == '-' or s == '&' or s=="\“" or s=="\”"\
#            or s == '<' or s == '>' or s == '(' or s == ')' \
#            or s == '[' or s == ']' or s == '{' or s == '}' or s == 'nbsp' or s == 'nbsp10' or s == '3.6' or s == 'about' or s == 'there' \
#            or s == "see" or s == "can" or s == "U" or s == "L" or s == "in" or s == ";" or s == "a" or s == "0144" \
#            or s == "\n" or s == "our" or s=="的" or s=="我们" or s=="要" or s=="自己" or s=="之"or s=="将"  \
#            or s=="后" or s=="应" or s=="到" or s=="某" or s=="后" or s=="个" or s=="是" or s=="位" or s=="新" \
#            or s=="一" or s=="两" or s=="在" or s=="中" or s=="或" or s=="有" or s=="更" or s=="好"

# 判断是否为停用词
def stop_word(s):
    with open("stopword.txt", 'r') as f:
        all_stopword = f.readlines()
        # print(all_stopword)
        for word in all_stopword:
            word = word.strip("\n")
            if s == word:
                return True
        return False
# 文档类别代码编号
ClassCode = ['C000007', 'C000008', 'C000010', 'C000013', 'C000014', 'C000016', 'C000020', 'C000022', 'C000023', 'C000024']

# 参数设置
Max_Documentcount_Eachclass = 300  # 每一个类的最大文件数
Train_DocumentCount_Eachclass = 250  # 每个类别选取250篇文档, 这样10类共选取了250*10=2000个文档作为训练集
# Test_DocumentCount_Eachclass = 20
Featurecount_Eachclass = 500  # 每类中选择的特征数量

# 构建每个类别的词Set
# 分词后的文件路径

textCutBasePath = sys.path[0] + "/SogouCSeg/"
ReadFilePathPrefix = os.path.dirname(os.path.abspath('__file__')) + "/SogouCSeg/"

#################################### 以下计算CHI, 选出特征词 #################################################
# 构建每个类别的词集合
def buildItemSets(classDocCount):
    """    
    :param classDocCount: 选择每个类别下文档的数量, 用于训练集合和测试集 
    :return: 
    """
    termDic = dict()
    # 每个类别下的文档集合用list<set>表示, 每个set表示一个文档，整体用一个dict表示
    termClassDic = dict()
    for eachclass in ClassCode:
        currClassPath = textCutBasePath + eachclass + "/"
        eachClassWordSets = set()
        eachClassWordList = list()
        for i in range(classDocCount):
            eachDocPath = currClassPath + str(i) + ".seg"
            eachFileObj = open(eachDocPath, 'r')
            eachFileContent = eachFileObj.read()
            eachFileWords = eachFileContent.split(" ")
            eachFileSet = set()
            for eachword in eachFileWords:
                # 判断是否是停止词, 如果是则丢掉
                stripeachword = eachword.strip(" ")
                if not stop_word(eachword) and len(stripeachword) > 0:
                    eachFileSet.add(eachword)
                    eachClassWordSets.add(eachword)
            eachClassWordList.append(eachFileSet)
            # print(eachFileSet)
        termDic[eachclass] = eachClassWordSets
        termClassDic[eachclass] = eachClassWordList
    # termDic={类别1：set（[类别1的所有文档的分词], .......,类别10：set（[类别10的所有文档的分词]）}
    # termClassDic={类别1：[set([该类中文档0的所有分词]), ......, set([该类中文档n的所有分词])], ........,类别10：[set([该类中文档0的所有分词]), ......, set([该类中文档n的所有分词])] }
    return termDic, termClassDic

# 对得到的两个词典进行计算，可以得到a b c d 值
# 开方计算公式, 对卡方检验所需的 a b c d 进行计算
def ChiCalc(a, b, c, d):
    '''    
    :param a: 在这个分类下包含这个词的文档数量
    :param b: 不在该分类下包含这个词的文档数量
    :param c: 在这个分类下不包含这个词的文档数量
    :param d: 不在该分类下不包含这个词的文档数量
    :return:  CHI值 
    '''
    # result = float(pow((a * d - b * c), 2)) / float((a + c) * (a + b) * (b + d) * (c + d))
    result = float(pow((a * d - b * c), 2)) / float((a + b) * (c + d))
    return result


# 用开方统计选取特征
def featureSelection_CHI(termDic, termClassDic, K):
    """    
    :param termDic: {类别1：set（[类别1的所有文档的分词], .......,类别10：set（[类别10的所有文档的分词]）}
    :param termClassDic: {类别1：[set([该类中文档0的所有分词]), ......, set([该类中文档n的所有分词])],........}
    :param K: 为每个类别选取的特征个数，比如K=500,, 那么对于10类来说,一共有10*500=5000个特征词, 经过去重后<=5000
    :return: termCountDic：每个类别下的K个特征词及其chi值构成的字典, 形如{类别1：{w1: chi, w2: chi2, ......, wk:chik},......}
    """
    termCountDic = dict()
    for key in termDic:
        classWordSets = termDic[key]
        classTermCountDic = dict()
        for eachword in classWordSets:  # 对某个类别下的每一个单词的 a b c d 进行计算
            a = 0
            b = 0
            c = 0
            d = 0
            for eachclass in termClassDic:
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
            # print("a+c:"+str(a+c)+"b+d"+str(b+d))
            eachwordcount = ChiCalc(a, b, c, d)
            # print(eachwordcount)
            classTermCountDic[eachword] = eachwordcount
        # 对生成的计数进行排序选择前K个
        # 这个排序后返回的是元组的列表
        sortedClassTermCountDic = sorted(classTermCountDic.items(), key=lambda d: d[1], reverse=True)
        # count = 0
        subDic = dict()
        for i in range(K):
            subDic[sortedClassTermCountDic[i][0]] = sortedClassTermCountDic[i][1]
        termCountDic[key] = subDic
    return termCountDic  # termCountDic={类别1：{w1: chi, w2: chi2, ......, wk:chik}, ......, 类别10：{w1: chi, w2: chi2, ......, wk:chik}}


def writeFeatureToFile(termCountDic, featurefileName, flag="CHI"):
    """    
    :param termCountDic: 每个类别下的K个特征词及其chi值构成的字典, 形如{类别1：{w1: chi, w2: chi2, ......, wk:chik},......}
    :param featurefileName: 保留特征词的文件名
    :param flag: "CHI" or "IG",使用“CHI”说明使用开方统计来提取特征, 使用“IG”说明使用信息增益来提取特征
    """
    featureSet = set()
    if flag == "CHI":
        for key in termCountDic:
            for eachkey in termCountDic[key]:  # 将各类的词,放到一个集合中，起到去重作用
                featureSet.add(eachkey)  # *******注意：集合中的元素是唯一的，所以使用集合有去除重复词的作用*********
        # featureSet=set([w1, w2, w3, ......,wn]) 其中wn是选出来的特征词，可能来自所有类
    if flag == "IG":
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

# ....................以下用信息增益选取特征...........................
def featureSelection_IG(termDic, termClassDic, K):
    """    
    :param termDic: {类别1：set（[类别1的所有文档的分词], .......,类别10：set（[类别10的所有文档的分词]）}
    :param termClassDic: {类别1：[set([该类中文档0的所有分词]), ......, set([该类中文档n的所有分词])],........}
    :param K: 特征维度
    :return: termCountDic：每个类别下的K个特征词及其chi值构成的字典, 形如{类别1：{w1: chi, w2: chi2, ......, wk:chik},......}
    """
    C = len(ClassCode)
    number_eachclass = float(Train_DocumentCount_Eachclass)
    Nall = float(C * number_eachclass)
    all_class_set = set()
    classTermCountDic = dict()

    for key in termDic:
        classWordSets = termDic[key]
        for eachword in classWordSets:
            all_class_set.add(eachword)
    for eachword in all_class_set:  # 对某个类别下的每一个单词的 a b c d 进行计算
        H_C_t = 0
        H_C_not_t = 0
        a = 0
        b = 0
        c = 0
        d = 0
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



################################### 以下计算TFIDF作为特征的权重值 ##################################################################


def readFeature(featurefileName):
    featureFile = open(featurefileName, 'r')
    featureContent = featureFile.read().split('\n')
    featureFile.close()
    feature = list()
    for eachfeature in featureContent:
        eachfeature = eachfeature.split(" ")
        if (len(eachfeature) == 2):
            feature.append(eachfeature[1])
    # print(feature)
    return feature  # feature=[w1, w2, w3,......,wn], 其中wn是特征词


# 读取所有类别的训练样本到字典中,每个文档是一个list
def readFileToList(textCutBasePath, ClassCode, Train_DocumentCount_Eachclass):
    '''    
    :param textCutBasePath: 分词文件所在的路径
    :param ClassCode: 类别列表
    :param Train_DocumentCount_Eachclass: 每一类中取的训练文档数量
    :param Test_DocumentCount_Eachclass: 每一类中取的测试文档数量
    :return: train_dic, test_dic={类别1：[[本类中文档0的词],[本类中文档1的词],......,[本类中文档n的词]],......., 类别10：[[本类中文档0的词],[本类中文档1的词],......,[本类中文档n的词]]}
    '''
    dic = dict()
    train_dic = dict()
    test_dic = dict()
    # count_class = 0
    for eachclass in ClassCode:
        currClassPath = textCutBasePath + eachclass + "/"
        eachclasslist = list()
        #########################################
        #对每一类下的文件数量进行统计
        count = 0
        for i in range(Max_Documentcount_Eachclass):
            f = currClassPath + str(i) + ".seg"
            if os.path.exists(f):
                count += 1
        Test_DocumentCount_Eachclass = count-Train_DocumentCount_Eachclass
        print("count of "+eachclass+":"+str(count)+"     count of test documents:"+str(Test_DocumentCount_Eachclass))
        ##########################################
        for i in range(Train_DocumentCount_Eachclass + Test_DocumentCount_Eachclass):
            eachfile = open(currClassPath+str(i)+".seg")
            eachfilecontent = eachfile.read()
            eachfilewords = eachfilecontent.split(" ")
            eachclasslist.append(eachfilewords)
            # print(eachfilewords)
        dic[eachclass] = eachclasslist
    for eachclass in ClassCode:
        train_dic[eachclass] = dic[eachclass][:Train_DocumentCount_Eachclass]
        test_dic[eachclass] = dic[eachclass][Train_DocumentCount_Eachclass:]
    return train_dic, test_dic  #train_dic/test_dic={类别1：[[本类中文档0的词],[本类中文档1的词],......,[本类中文档n的词]],......., 类别10：[[本类中文档0的词],[本类中文档1的词],......,[本类中文档n的词]]}




# 计算特征的逆文档频率
def featureIDF(dic, feature, dffilename):
    dffile = open(dffilename, "w")  # 新建文件，避免同名文件
    dffile.close()
    dffile = open(dffilename, "a")
    totaldoccount = 0
    idffeature = dict()
    dffeature = dict()
    for eachfeature in feature:
        docfeature = 0
        for key in dic:
            totaldoccount = totaldoccount + len(dic[key])
            classfiles = dic[key]
            for eachfile in classfiles:
                if eachfeature in eachfile:
                    docfeature = docfeature + 1
        # 计算特征的逆文档频率
        featurevalue = math.log(float(totaldoccount) / (docfeature + 1))  # *****注意：分母加1, 避免分母为0******
        dffeature[eachfeature] = docfeature
        # 写入文件，特征的文档频率
        dffile.write(eachfeature + " " + str(docfeature) + "\n")  # w1 文档频率 \n w2 文档频率 \n w3 文档频率......
        # print(eachfeature+" "+str(docfeature))
        idffeature[eachfeature] = featurevalue
    dffile.close()
    return idffeature  # idffeature={w1: idf,.......,wn:idf}


# 计算Feature's TF-IDF 值
def TFIDFCal(feature, dic, idffeature, svmfilename, commonfilename=None):
    """    
    :param feature:已经选出特征词 
    :param dic: 训练字典或测试字典
    :param idffeature: 特征词在整个训练集或测试集中的出现次数的统计
    :param svmfilename: 用于libsvm的稀疏数据格式, 如（1， 1:0.1， 20:0.99，......，4102:0.6）
    :param commonfilename: 一般的密集向量表示格式，如（1， 2， 3，......4902）
    :return: 得到两个文件
    """
    file = open(svmfilename, 'w')
    file.close()
    file = open(svmfilename, 'a')

    commonfile = open(commonfilename, 'w')
    commonfile.close()
    commonfile = open(commonfilename, 'a')

    for key in dic:
        classFiles = dic[key]
        # 谨记字典的键是无序的
        classid = ClassCode.index(key)  # 得到键对应的索引,即标签
        for eachfile in classFiles:
            # 对每个文件进行特征向量转化
            file.write(str(classid) + " ")
            commonfile.write(str(classid) + " ") #密集型向量格式格式的文件，用于自己搭建的分类器
            for i in range(len(feature)):
                if feature[i] in eachfile:
                    currentfeature = feature[i]
                    featurecount = eachfile.count(feature[i])
                    tf = float(featurecount) / (len(eachfile))
                    # 计算逆文档频率
                    featurevalue = idffeature[currentfeature] * tf
                    file.write(str(i + 1) + ":" + str(featurevalue) + " ")
                    commonfile.write(str(featurevalue) + " ")
                else:
                    commonfile.write(str(0) + " ")
            file.write("\n")
            commonfile.write("\n")



if __name__ == "__main__":
    Flag = "CHI"
    # 调用buildItemSets
    # buildItemSets形参表示每个类别的文档数目,在这里训练模型时每个类别取前200个文件
    print('...................特征选择....................')
    termDic, termClassDic = buildItemSets(Train_DocumentCount_Eachclass)
    if Flag == "CHI":
        print('..................使用CHI选取特征.................')
        termCountDic = featureSelection_CHI(termDic, termClassDic, K=Featurecount_Eachclass)
        writeFeatureToFile(termCountDic, "Feature.txt")
        print('..................已选取特征.................')
        train_dic, test_dic = readFileToList(textCutBasePath, ClassCode, Train_DocumentCount_Eachclass)
        feature = readFeature("Feature.txt")
        print("length of feature: ", len(feature))

        print (".................生成训练集...............")
        idffeature_trainset = featureIDF(train_dic, feature, "dffeature_train.txt")
        TFIDFCal(feature, train_dic, idffeature_trainset, "train.svm", "train.txt")
        print(".................已获得训练集..............")

        print (".................生成测试集...............")
        idffeature_testset = featureIDF(test_dic, feature, "dffeature_test.txt")
        TFIDFCal(feature, test_dic, idffeature_testset, "test.svm", "test.txt")
        print(".................已获得测试集..............")

    if Flag == "IG":
        print('..................使用IG选取特征.................')
        termCountDic = featureSelection_IG(termDic, termClassDic, K=4840)  #('length of feature: ', 4840)
        writeFeatureToFile(termCountDic, "IG_Feature.txt", flag="IG")
        print('..................已选取特征.................')
        train_dic, test_dic = readFileToList(textCutBasePath, ClassCode, Train_DocumentCount_Eachclass)
        feature = readFeature("IG_Feature.txt")
        print("length of feature: ", len(feature))

        print (".................生成训练集...............")
        idffeature_trainset = featureIDF(train_dic, feature, "IG_dffeature_train.txt")
        TFIDFCal(feature, train_dic, idffeature_trainset, "IG_train.svm", "IG_train.txt")
        print(".................已获得训练集..............")

        print (".................生成测试集...............")
        idffeature_testset = featureIDF(test_dic, feature, "IG_dffeature_test.txt")
        TFIDFCal(feature, test_dic, idffeature_testset, "IG_test.svm", "IG_test.txt")
        print(".................已获得测试集..............")



"""
.使用CHI提取的特征：
    使用步骤：
    1. 安装libsvm-tools   
    2. 接下来可以开始训练了：
       （1）可以使用svm-easy命令进行训练和预测，一步到位，具体命令：svm-easy train.svm test.svm
        (2) 也可以分开进行,具体如下：
            A. 对train.svm文件数据进行缩放到[0,1]区间
        
               svm-scale -l 0 -u 1 train.svm > trainscale.svm
            
            B. 对test.svm文件数据进行缩放到[0,1]区间
            
               svm-scale -l 0 -u 1 test.svm > testscale.svm
            
            C. 对trainscale.svm 文件进行模型训练
            
               svm-train -s 1 trainscale.svm trainscale.model
            
            D. 对testscale.svm 文件进行模型预测，得到预测结果，控制台会输出正确率
            
               svm-predict testscale.svm trainscale.model testscale.result
               
            实验：
                 (1)每一类训练文档：200  每一类测试数据：20   每一类特征维度：1000
                     交叉验证： python libsvm-3.22/tools/grid.py trainscale.svm ---->(best c:32, g:0.0078125)
                     svm-train -c 32 -g 0.0078125 trainscale.svm    其中：-c是惩罚因子, -g是RBF的gramma值
                     svm-predict testscale.svm trainscale.svm.model testscale.svm.predict (81.5%)
                 
                 (2)每一类训练文档：250  每一类测试数据：每一类的可以自己计算, 总共466个   每一类特征维度：500
                    Accuracy = 84.7639% (395/466) (classification)  (换了停用词表)


.使用IG提取的特征：
             
            A. 对train.svm文件数据进行缩放到[0,1]区间
        
               svm-scale -l 0 -u 1 IG_train.svm > IG_trainscale.svm
            
            B. 对test.svm文件数据进行缩放到[0,1]区间
            
               svm-scale -l 0 -u 1 IG_test.svm > IG_testscale.svm
            
            C. 对trainscale.svm 文件进行模型训练
            
               svm-train -s 1 IG_trainscale.svm IG_trainscale.model
            
            D. 对testscale.svm 文件进行模型预测，得到预测结果，控制台会输出正确率
            
               svm-predict IG_testscale.svm IG_trainscale.model IG_testscale.result
               
            result:  Accuracy = 84.9785% (396/466) (classification) (换了停用词表)


"""











