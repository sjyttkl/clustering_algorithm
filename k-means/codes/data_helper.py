# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     data_helper
   email:         songdongdong@jd.com
   Author :       songdongdong
   date：          2019/7/16
   Description :  
==================================================
"""
__author__ = 'songdongdong'

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

# Open PhantomJS
driver = webdriver.PhantomJS()
# driver = webdriver.Firefox()
wait = ui.WebDriverWait(driver, 10)


# Get the Content of 5A tourist spots
def getInfobox(entityName, fileName):
    try:
        # create paths and txt files
        print( u'文件名称: ', fileName)
        info = codecs.open(fileName, 'w', 'utf-8')

        # locate input  notice: 1.visit url by unicode 2.write files
        # Error: Message: Element not found in the cache -
        #       Perhaps the page has changed since it was looked up
        # 解决方法: 使用Selenium和Phantomjs
        print(u'实体名称: ', entityName.rstrip('\n'))
        driver.get("http://baike.baidu.com/")
        elem_inp = driver.find_element_by_xpath("//form[@id='searchForm']/input")
        elem_inp.send_keys(entityName)
        elem_inp.send_keys(Keys.RETURN)
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

    # Main function


def main():
    # By function get information
    path = "BaiduSpider\\"
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
        getInfobox(entityName, fileName)
        num = num + 1
    print('End Read Files!')
    source.close()
    driver.close()


if __name__ == '__main__':
    main()
