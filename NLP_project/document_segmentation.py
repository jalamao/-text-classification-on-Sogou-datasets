# coding:utf-8
__author__ = "maotingyun"
import jieba
import io
import os

"""
备注：

(1)对于语料SogouC的预处理：格式转换，在每个文件夹下, 打开终端，使用enca批处理：enca -x utf-8 * 
(2)格式转换结束后，每个文件夹下可能有未成功转换的文件，把这种文件叫做坏文件
(3)文件分词，存放到SogouCSeg的对应文件夹中，分词期间，要过滤掉坏文件，只对好文件进行分词，每类中得到的分词文件数量<=300

"""

FolderList = ['C000007', 'C000008', 'C000010', 'C000013', 'C000014', 'C000016', 'C000020', 'C000022','C000023', 'C000024']
FolderTextCount = 300   # 每个类中原始文件数量（在这里只选择搜狗预料库中的每个分类下的300个文本进行分词处理）

ReadFilePathPrefix = os.path.dirname(os.path.abspath('__file__'))+"/SogouC/ClassFile/"  # 原始语料路径

WriteFilePathPrefix = os.path.dirname(os.path.abspath('__file__')) + "/SogouCSeg/"   # 语料被分词后的保存路径


# 对每个类别下的文档进行分词处理
def cut_texts_eachclass():
    """
    备注：下载的数据是GB2312格式的, 在ubuntu下需要转换成utf-8格式，使用的工具是enca, 该工具的优点是可以使用该格式转换工具对文件夹下的所有文件进行批处理 
        但是, 处理完的文件中, 有个别文件是乱码的， 使用jieba分词时, 需要滤掉这些坏掉的文件, 在该函数中，使用了try/finally语句来实现过滤坏文件.
    """
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
            finally: # finally相当于continue，跳过本次循环
                file_count +=1
                resultfile.close()
    print("......................OK....................")

################################################################################

if __name__ == "__main__":

    cut_texts_eachclass()