# 设计师:Pan YuDong
# 编写者:God's hand
# 时间:2022/6/30 15:43
import re
import csv
import pandas as pd
from selenium import webdriver
from selenium.webdriver.edge.service import Service
from bs4 import BeautifulSoup

def update_userInfo(filepath, row):
    '''
    :param filepath: path of userInfo file
    :param row: the row we want to update
    :return:
    '''
    df = pd.read_csv(filepath)
    df.loc[row, 'has_finish'] = '1'
    df.to_csv(filepath, index=False)

def update_dataInfo(filepath, data_batch, is_exist=False):
    '''
    :param filepath: path of userInfo file
    :param data_batch: list of data we want to write
    :param is_exist: is the file exist
    :return:
    '''
    way = 'a' if is_exist else 'w'
    with open(filepath, way, newline='') as csvfile:
        writer = csv.writer(csvfile)
        # If it is the first time, write column names first
        if not is_exist:
            writer.writerow(["user", "URL", "has_finish", "repos", "stars"])
        # write a batch of data
        writer.writerows(data_batch)

def spider_user_info(source_file, target_file):
    '''
       :param source_file: List of projects to crawl user info
       :param target_file: Crawled saved data
       :return:
       '''

    # top 30 projects in machine learning
    df = pd.read_csv(source_file)
    top30_projects = df['project']
    finish_list = df['has_finish']

    urls = ['https://github.com/' + project + '/stargazers' for project in top30_projects]


    edge_options = webdriver.EdgeOptions()
    edge_options.add_argument('--headless')
    s = Service('D:\BrowserDriver\msedgedriver.exe')
    browser = webdriver.Edge(service=s, options=edge_options)

    project_cnt = 0
    for url in urls:
        if finish_list[project_cnt] == 1:
            project_cnt += 1
            continue

        page = 1
        data_batches = []
        while True:
            browser.get(url + '?page=' + str(page))

            html = browser.page_source
            bf = BeautifulSoup(html, 'lxml')
            tag_user = bf.find_all(name=re.compile('a'), attrs={'data-hovercard-type':'user'})

            data_batch = []
            for i in range(len(tag_user)):
                tag_text = tag_user[i].text
                if tag_text != '':
                    data_batch.append([tag_text, 'https://github.com/' + tag_text, '0', '0', '0'])
            print("data_batch:", data_batch)

            # find next arrow
            tag_next = bf.find(name=re.compile('a'), attrs={'rel':'nofollow'}, string="Next")

            # if next arrow not found in the page
            print("tag_next:", tag_next)
            if not tag_next:
                break

            page += 1
            data_batches += data_batch

        if project_cnt == 0:
            update_dataInfo(target_file, data_batches, is_exist=False)
        else:
            update_dataInfo(target_file, data_batches, is_exist=True)

        update_userInfo(source_file, project_cnt)
        project_cnt += 1

source_file = '../Resource/top30_projects.csv'
target_file = 'github_users/raw_users.csv'
spider_user_info(source_file, target_file)