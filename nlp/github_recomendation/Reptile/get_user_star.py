# 设计师:Pan YuDong
# 编写者:God's hand
# 时间:2022/6/30 18:34
import asyncio
import aiohttp
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.edge.service import Service
from bs4 import BeautifulSoup
from fake_useragent import UserAgent


async def fetch(session, url, headers):
    '''
    :param session: session process
    :param url: the link to fetch web data
    :param headers: request page headers
    :return:
    '''
    try:
        async with session.get(url, headers=headers, timeout=40) as response:
            print("response.status:", response.status)
            if response.status != 200:
                return None
            return await response.text()
    except Exception as e:
        print(e)
        time.sleep(3)
        print("continue")
        return None

async def get_results(url, headers):
    '''
    :param url: the link to fetch web data
    :param headers: request page headers
    :return:
    '''
    async with aiohttp.ClientSession() as session:
        html = await fetch(session, url, headers)
        if not html:
            return 0, 0

        bf1 = BeautifulSoup(html, 'lxml')
        result = bf1.find_all(class_='Counter')
        # count the number of layouts
        layout_cnt = len(result)
        if layout_cnt < 4:
            repo = 0
            star = 0
        else:
            repo = result[0].text
            star = result[3].text
        print(f"url:{url}, repo:{repo}, star:{star}")
        return repo, star


def update_userInfo(filepath, start, end, repos, stars):
    '''
     :param filepath: path of userInfo file
     :param start: the start row we want to update
     :param end: the end row we want to update
     :param repos: list of repos
     :param stars: list of stars
     :return:
    '''
    df = pd.read_csv(filepath)
    print("repos:", repos)
    print("stars:", stars)

    print("start:", start)
    print("end:", end)
    print("len(repos):", len(repos))

    df.loc[start:end, 'has_finish'] = '1'
    df.loc[start:end, 'repos'] = repos
    df.loc[start:end, 'stars'] = stars

    df.to_csv(filepath, index=False)
    print("done....")

def spider_user_star_aio(filepath):
    '''
       :param filepath: path of userInfo file
    '''

    df = pd.read_csv(filepath)
    urls = df['URL']
    finish_list = df['has_finish']

    # edge_options = webdriver.EdgeOptions()
    # edge_options.add_argument('--headless')
    # s = Service('D:\BrowserDriver\msedgedriver.exe')
    # browser = webdriver.Edge(service=s, options=edge_options)

    user_cnt = 0
    req_cnt = 0
    token_cnt = 0
    N = 30
    repos = []
    stars = []
    ua = UserAgent()
    total_users = len(urls)
    while user_cnt <= total_users:
        if finish_list[user_cnt] == 1:
            user_cnt += 1
            continue

        # need to supplement
        df_token = pd.read_csv('../Resource/tokens.csv')
        token_list = df_token['token'].tolist()

        headers = {'User-Agent': ua.random,
                   'Authorization': f'token {token_list[token_cnt % len(token_list)]}',}

        loop = asyncio.get_event_loop()
        tasks = [loop.create_task(get_results(url, headers)) for url in urls[user_cnt:user_cnt + N]]
        loop.run_until_complete(asyncio.wait(tasks))
        for task in tasks:
            repo, star = task.result()
            repos.append(repo)
            stars.append(star)

        # when user_cnt reaches times of 30, write into csv file
        update_userInfo(filepath, user_cnt, user_cnt + N - 1, repos, stars)
        repos.clear()
        stars.clear()

        user_cnt += N
        req_cnt += N
        if req_cnt >= 4800:
            token_cnt += 1

def spider_user_star_base(filepath):
    '''
       :param filepath: path of userInfo file
    '''

    df = pd.read_csv(filepath)
    urls = df['URL']
    finish_list = df['has_finish']

    edge_options = webdriver.EdgeOptions()
    edge_options.add_argument('--headless')
    s = Service('D:\BrowserDriver\msedgedriver.exe')
    browser = webdriver.Edge(service=s, options=edge_options)

    user_cnt = 0
    repos = []
    stars = []
    for url in urls:
        if finish_list[user_cnt] == 1:
            user_cnt += 1
            continue

        browser.get(url)
        html = browser.page_source
        browser.close()
        bf1 = BeautifulSoup(html, 'lxml')
        result = bf1.find_all(class_='Counter')

        # count the number of layouts
        layout_cnt = len(result)

        if layout_cnt < 4:
            repo = 0
            star = 0
        else:
            repo = result[0].text
            star = result[3].text

        print(f"url:{url}, repo:{repo}, star:{star}")

        repos.append(repo)
        stars.append(star)
        user_cnt += 1

        # when user_cnt reaches times of 30, write into csv file
        if user_cnt % 30 == 0:
            update_userInfo(filepath, user_cnt - len(repos), user_cnt - 1, repos, stars)
            repos.clear()
            stars.clear()


spider_user_star_aio('github_users/raw_users.csv')
