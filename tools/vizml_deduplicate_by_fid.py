from time import time
from tqdm import tqdm
import pandas as pd
import numpy as np
import multiprocessing
import datetime
import pickle
import json
import csv
import sys
import os
sys.path.insert(0, '..')
from helpers.logger import logger
from feature_extraction.type_detection import detect_field_type, data_type_to_general_type, data_types, general_types


# 未完成，因发现使用VizML提供的deduplicate_charts也可以，效率也不算太低
# 应该只记录了下原始205G的数据里属于同个user的所有fid

class deduplicator(object):
	def __init__(self):
		# if not os.path.exists('./logs/'): os.mkdir('./logs/')
		# log_suffix = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
		# log_file = './logs/deduplicate_by_fid_' + log_suffix + '.txt'
		# self.logger			= logger(log_file, {'Mission': 'deduplicate vizml per user'})
		self.input_file		= '../data/vizml_ded/original_tsv/plot_data_with_all_fields_and_header.tsv'
		self.output_file	= '../data/vizml_ded/original_tsv/deduplicated_one_per_user.tsv'
		self.users_file		= '../data/vizml_ded/original_tsv/users_info_before_ded.json'
		self.table_headers	= ['fid', 'chart_data', 'layout', 'table_data']
		self.chunk_size		= 1000
		self.parallelize	= True
		self.users	= {}
		self.charts_without_data = 0


	def main_process(self):
		# raw_df_chunks = self.load_raw_data()
		# self.enumerate_users(raw_df_chunks)	# 已完成，直接去读取users文件就可以了
		with open(self.users_file, 'r') as f:
			self.users = json.load(f)

		print(len(self.users.keys()))

		print('Meindert' in self.users.keys())
		if 'Meindert' in self.users.keys():
			for fid in self.users['Meindert'].keys():
				print(fid)

		# load_users_info
		# search per user


	def load_raw_data(self):
		self.logger.log('Loading raw data from ' + self.input_file)
		df = pd.read_csv(
			self.input_file,
			sep='\t',
			error_bad_lines=False,
			chunksize=self.chunk_size,
			encoding='utf-8'
		)
		return df


	def enumerate_users(self, raw_df_chunks):
		self.logger.log('Start traversing chunks to find all users')
		for chunk_id, chunk in tqdm(enumerate(raw_df_chunks)):
			for row_id, row in chunk.iterrows():
				chart_index	= chunk_id * self.chunk_size + row_id		# start from 0
				fid			= row.fid
				chart_data	= json.loads(row.chart_data)
				layout		= json.loads(row.layout)
				table_data	= json.loads(row.table_data)
				if not (bool(chart_data) and bool(table_data)):
					self.logger.log('still empty data row[{}]: {}'.format(chart_index, fid))
					self.charts_without_data += 1
					continue
				user = fid.split(':')[0]
				if not user in self.users: self.users[user] = [{fid: chart_index}]
				else: self.users[user].append({fid: chart_index})

		self.logger.log('error rows num: {}'.format(self.charts_without_data))
		users_info = json.dumps(self.users)
		with open(self.users_file, 'w') as f:
			f.write(users_info)
		f.close()



if __name__ == '__main__':
	d = deduplicator()
	d.main_process()