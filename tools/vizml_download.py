# 很多 Temporal 的数据不一样，网站上的数据还在update

from tqdm import tqdm
import pandas as pd
import requests
import os

requests.adapters.DEFAULT_RETRIES = 3
ses = requests.session()
ses.keep_alive = False


dataset = 'vizml_1k'
raw_data_folder = '../data/%s/raw_data/' % dataset
# save_folder = '../data/%s/png/' % dataset
# save_folder = '../data/%s/json/' % dataset
save_folder = '../data/%s/csv/' % dataset
chunk_files = os.listdir(raw_data_folder)

if not os.path.exists(save_folder):
    os.mkdir(save_folder)

# img_not_exist = open('../data/%s/img_not_exist.txt' % dataset, 'w+')
# img_exist     = open('../data/%s/img_exist.txt' % dataset, 'w+')

chart_index = -1
for chunk_file in chunk_files:
    chunk = pd.read_csv(raw_data_folder + chunk_file)
    for i, chart_obj in tqdm(chunk.iterrows()):
        chart_index += 1
        fid = chart_obj.fid
        # url = 'https://chart-studio.plotly.com/~{0}/{1}.png'.format(*fid.split(':'))
        # url = 'https://chart-studio.plotly.com/~{0}/{1}.json'.format(*fid.split(':'))
        url = 'https://chart-studio.plotly.com/~{0}/{1}.csv'.format(*fid.split(':'))
        # response = requests.get(url)
        response = ses.get(url)
        if response.status_code == 200:
            # file_name = save_folder + '{0}_{1}.png'.format(*fid.split(':'))
            # file_name = save_folder + '{0}_{1}.json'.format(*fid.split(':'))
            file_name = save_folder + '{0}_{1}.csv'.format(*fid.split(':'))
            with open(file_name, 'wb') as f:
                f.write(response.content)
            # img_exist.write(fid)
            # break
        elif response.status_code == 404:
            # img_not_exist.write(fid)
            pass
        else:
            print(chart_index, fid)
            print(response.status_code)
        response.close()
    # break

# img_exist.close()
# img_not_exist.close()