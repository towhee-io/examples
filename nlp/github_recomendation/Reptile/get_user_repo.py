# 设计师:Pan YuDong
# 编写者:God's hand
# 时间:2022/6/22 15:47
import asyncio
import random
import aiohttp
import pandas as pd
import csv
import time
from fake_useragent import UserAgent
from Utils import script

def get_userInfo(filepath):
    '''
    :param filepath: path of userInfo file
    :return: username and has_finish list
    '''
    df = pd.read_csv(filepath)
    usernames = df['user']
    finish_list = df['has_finish']
    star_list = df['stars']
    return usernames, finish_list, star_list


async def fetch(session, url, headers):
    '''
    :param session: path of userInfo file
    :param url: the link to fetch web data
    :param headers: request page headers
    :return:
    '''
    try:
        async with session.get(url, headers=headers, timeout=40) as response:
            print("response.status:", response.status)
            if response.status != 200:
                return None
            return await response.json()
    except:
        print("requests speed so high,need sleep!")
        time.sleep(5)
        print("continue")
        return None


async def get_results(url, username, headers):
    '''
    :param url: the link to fetch web data
    :param username: target user we want to search
    :param headers: request page headers
    :return:
    '''
    async with aiohttp.ClientSession() as session:
        result = await fetch(session, url, headers)
        num_projects = 0 if not result else len(result)
        item_list = []
        # print("num_projects:", num_projects)
        # print("result:", result)
        for i in range(num_projects):
            project_name = result[i]['full_name']
            num_star = result[i]['stargazers_count']
            num_fork = result[i]['forks_count']
            item_list.append([username, project_name, str(num_star), str(num_fork), '1'])
        if num_projects > 0:
            print(f"num_projects = {num_projects}")
        return item_list


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
            writer.writerow(["user", "project", "star", "fork", "has_star"])
        # write a batch of data
        writer.writerows(data_batch)


def spider_user_repo(source_file, target_file, index):
    '''
    :param source_file: List of users to crawl data
    :param target_file: Crawled saved data
    :param headers: request page headers
    :return:
    '''
    usernames, finish_list, star_list = get_userInfo(source_file)
    print("username:", usernames)
    user_cnt = 0
    ua = UserAgent()

    # need to supplement
    df_token = pd.read_csv('../Resource/tokens.csv')
    token_list = df_token['token'].tolist()

    for username in usernames:
        print(f"Crawling data for user:{username}")
        # if this user has finished, skip it
        if finish_list[user_cnt] == 1:
            user_cnt += 1
            continue

        items_list = []
        req_num = star_list[user_cnt] // 100 + 1
        urls = ['https://api.github.com/users/{}/starred?page={}&per_page=100'.format(username, i + 1)
                for i in range(req_num)]
        headers = {'User-Agent': ua.random,
                   'Authorization': f'token {token_list[random.randint(0, len(token_list) - 1)]}',
                   'cookie': 'login information',
                   'Content-Type': 'application/json',
                   'Accept': 'application/json'}

        loop = asyncio.get_event_loop()
        tasks = [loop.create_task(get_results(url, username, headers)) for url in urls]
        loop.run_until_complete(asyncio.wait(tasks))
        for task in tasks:
            items = task.result()
            if len(items) > 0:
                items_list += items
        # print("items_list:", items_list)

        update_userInfo(source_file, user_cnt)

        order = user_cnt // 1000
        target_path = target_file + f'data{index * 10000 + order * 1000 + 1}-{index * 10000 + (order + 1) * 1000}.csv'
        print("target_path:", target_path)
        if user_cnt % 1000 == 0:
            update_dataInfo(target_path, items_list, is_exist=False)
        else:
            update_dataInfo(target_path, items_list, is_exist=True)

        user_cnt += 1


index = 2
source_file = 'github_users/filtered_user{}-{}W.csv'.format(index, index + 1)
target_file = 'data_result/filtered_data{}-{}W/'.format(index, index + 1)
spider_user_repo(source_file, target_file, index)
script.merge_data(target_file, index=index, fold_num=10000)


