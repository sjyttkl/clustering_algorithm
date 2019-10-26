# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     scrapy_baidu
   email:         songdongdong@jd.com
   Author :       songdongdong
   date：          2019/7/16
   Description :  爬取 baidu信息
==================================================
"""
__author__ = 'sjyttkl'

import time
import re
import os
import sys
import codecs
import shutil
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import selenium.webdriver.support.ui as ui
from selenium.webdriver.common.action_chains import ActionChains
options = webdriver.ChromeOptions()
# options.set_headless() #无头
driver = webdriver.Chrome(chrome_options=options)
ait = ui.WebDriverWait(driver, 10)
def getInfobox(entityName, fileName):
    try:
        # create paths and txt files
        print( u'文件名称: ', fileName)
        info = codecs.open(fileName, 'w', 'utf-8')
        print(u'实体名称: ', entityName.rstrip('\n'))
        elem_inp = driver.find_element_by_xpath("//form[@id='searchForm']/input")
        elem_inp.send_keys(entityName)
        time.sleep(2)
        # elem_inp.send_keys(Keys.RETURN)
        info.write(entityName.rstrip('\n') + '\r\n')  # codecs不支持'\n'换行
        time.sleep(2)

        # load content 摘要
        elem_value = driver.find_elements_by_xpath("//div[@class='lemma-summary']/div")
        for value in elem_value:
            print(value.text)
            info.writelines(value.text + '\r\n')
        time.sleep(2)

    except Exception as  e:  # 'utf8' codec can't decode byte
        print("Error: ", e)
    finally:
        print( '\n')
        info.close()

def main():
    path = "../data/BaiduSpider\\"
    if os.path.isdir(path):
        shutil.rmtree(path, True)
    os.makedirs(path)
    source = open("../data/Tourist_spots_5A_BD.txt", 'r',encoding="utf-8")
    num = 1
    for entityName in source:
        # entityName = unicode(entityName, "utf-8")
        if u'故宫' in entityName:  # else add a '?'
            entityName = u'北京故宫'
        name = "%04d" % num
        fileName = path + str(name) + ".txt"
        driver.get("http://baike.baidu.com/")
        getInfobox(entityName, fileName)
        num = num + 1
    print('End Read Files!')
    source.close()
    driver.close()


if __name__ == '__main__':
    main()
