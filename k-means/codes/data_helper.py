# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     data_helper
   email:         695492835@qq.com
   Author :       sjyttkl
   date：          2019/7/16
   Description :  
==================================================
"""
__author__ = 'sjyttkl'
import sys
import re
import io
import os
import shutil
import jieba
import jieba.analyse

# 导入自定义词典
jieba.load_userdict("../data/dict_baidu.txt")

def read_file_cut():
    # create path
    path = "..\\data\\BaiduSpider\\"
    respath = "..\\data\\BaiduSpider_Result\\"
    if os.path.isdir(respath):
        shutil.rmtree(respath, True) #来删除目录及目录内部的文件，可能会有文件残留
    os.makedirs(respath)

    num = 1
    while num <= 20:
        name = "%04d" % num
        fileName = path + str(name) + ".txt"
        resName = respath + str(name) + ".txt"
        source = open(fileName, 'r',encoding="utf-8")
        if os.path.exists(resName):
            os.remove(resName)
        result = io.open(resName, 'w', encoding='utf-8')
        line = source.readline()
        line = line.strip()

        while line != "":
            # line = unicode(line, "utf-8")
            seglist = jieba.cut(line, cut_all=False)  # 精确模式
            output = ' '.join(list(seglist))  # 空格拼接
            print(output)
            result.write(output + '\r\n')
            line = source.readline()
        else:
            print('End file: ' + str(num))
            source.close()
            result.close()
        num = num + 1
    else:
        print('End All')


def merge_file():
    path = "../data/BaiduSpider_Result\\"
    resName = "../data/BaiduSpider_Result.txt"
    if os.path.exists(resName):
        os.remove(resName)
    result = io.open(resName, 'w', encoding='utf-8')

    num = 1
    while num <= 20:
        name = "%04d" % num
        fileName = path + str(name) + ".txt"
        source = open(fileName, 'r',encoding="utf-8")
        line = source.readline()
        line = line.strip()

        while line != "":
            line = line.replace('\n', ' ')
            line = line.replace('\r', ' ')
            result.write(line + ' ')
            line = source.readline()
        else:
            print( 'End file: ' + str(num))
            result.write('\n')
            source.close()
        num = num + 1

    else:
        print('End All')
        result.close()
# Run function
if __name__ == '__main__':
    # read_file_cut()
    merge_file()#兼并文件