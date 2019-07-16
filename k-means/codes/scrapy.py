# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     scrapy
   email:         695492835@qq.com
   Author :       sjyttkl
   date：          2019/7/16
   Description :  
==================================================
"""
__author__ = 'sjyttkl'
# encoding=utf-8
import sys
import re
import codecs
import os
import shutil
import jieba
import jieba.analyse

# 导入自定义词典
jieba.load_userdict("dict_baidu.txt")

# Read file and cut
def read_file_cut():
    # create path
    path = "BaiduSpider\\"
    respath = "BaiduSpider_Result\\"
    if os.path.isdir(respath):
        shutil.rmtree(respath, True)
    os.makedirs(respath)

    num = 1
    while num <= 204:
        name = "%04d" % num
        fileName = path + str(name) + ".txt"
        resName = respath + str(name) + ".txt"
        source = open(fileName, 'r')
        if os.path.exists(resName):
            os.remove(resName)
        result = codecs.open(resName, 'w', 'utf-8')
        line = source.readline()
        line = line.rstrip('\n')

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


# Run function
if __name__ == '__main__':
    read_file_cut()
