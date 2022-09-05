# 设计师:Pan YuDong
# 编写者:God's hand
# 时间:2022/9/5 19:25
import pandas as pd
from Utils import preprocessing

source_file = 'raw_users.csv'
target_file = 'filtered_users.csv'

low_star = 10
high_star = 1000

preprocessing.transform_data(source_file, target_file)
preprocessing.clear_data(target_file, low_star, high_star)

