# 设计师:Pan YuDong
# 编写者:God's hand
# 时间:2022/6/30 21:09
import numpy as np
import pandas as pd


def transform_data(source_path, target_path):
    df = pd.read_csv(source_path)

    # delete rows where stars = 0 and repositories = 0
    df = df.drop(index=df.stars[df.stars == '0'].index)

    # filter columns repos and stars
    repos = df['repos'].tolist()
    stars = df['stars'].tolist()
    repos = [repo.split('\n')[-1] for repo in repos]
    stars = [star.split('\n')[-1] for star in stars]

    # covert str to int
    print("repos:", repos)
    print("stars:", stars)

    repo_list = []
    star_list = []

    for repo, star in zip(repos, stars):
        if 'k' in repo:
            repo_list.append(int(float(repo.split('k')[0]) * 1000))
        elif 'Repositories' in repo:
            repo_list.append(0)
        else:
            repo_list.append(int(repo))

        if 'k' in star:
            star_list.append(int(float(star.split('k')[0]) * 1000))
        elif 'Stars' in star:
            star_list.append(0)
        else:
            star_list.append(int(star))

    print("repo_list:", repo_list)
    print("star_list:", star_list)

    df['repos'] = repo_list
    df['stars'] = star_list
    df['has_finish'] = 0

    print("df:", df)

    # save file
    df.to_csv(target_path, index=False)


def clear_data(filepath, low_star, high_star):
    df = pd.read_csv(filepath)

    # delete rows where low_star <= stars = high_star
    df = df[(df.stars >= low_star) & (df.stars <= high_star)]

    df.to_csv(filepath, index=False)
