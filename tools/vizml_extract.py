
# 对于每个数据集，对每个维度预测x或y，以及type
# 对所有x和y的组合，只要长度相同，就根据y的type绘制图表
# 指标：组合准确率、召回率，type准确率，x和y的type相同的准确率
# 统计：outcomes中x和y相同的trace占比，各个trace的type占比，各个维度长度相同占比，是否只有同个trace长度相同

# 保存、读取pairwise特征
# 使用pairwise-level特征训练 vs 只用field-lvel特征训练
# 使用pairwise直接预测能否组合，以及trace type

# 写一个 look_the_data.py



####### pairwise feature extraction and save

from tqdm import tqdm
import pandas as pd
import numpy as np
import multiprocessing
import datetime
import pickle
import time
import json
import csv
import sys

sys.path.insert(0, '..')
from helpers.logger import logger
from feature_extraction.type_detection import detect_field_type, data_type_to_general_type
from feature_extraction.pairwise_field_features import extract_pairwise_field_features, all_pairwise_features_list
from feature_extraction.helpers import get_unique, parse


raw_data_file = '../data/vizml_ded/original_tsv/plot_data_with_all_fields_and_header_deduplicated_one_per_user.tsv'
pairwise_features_file = '../data/vizml_ded/features/pairwise_features.csv'
parallelize = True

log_suffix = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
log_file = 'logs/' + 'extract_pairwise_featuers_{}.txt'.format(log_suffix)
extract_logger = logger(log_file, {'Mission': 'extracting pairwise features for ded charts', 'Parallelize': parallelize})

MAX_FIELDS = 25
chunk_size = 1000
fids_exceeding_max_fields = []


def construct_chart_pairwise_features_df(pairwise_field_features):
	cols = [feature['name'] for feature in all_pairwise_features_list]
	df = pd.DataFrame(columns=cols)
	for one_field_pairwise_field_features in pairwise_field_features:
		for pairwise_features in one_field_pairwise_field_features:
			if pairwise_features['pair_exists']:
				df = df.append([pairwise_features], ignore_index=True)
	return df


def extract_in_a_chunk(chunk):
	cols = [feature['name'] for feature in all_pairwise_features_list]
	chunk_pairwise_features_df = pd.DataFrame(columns=cols)

	for chart_num, chart_obj in chunk.iterrows():
		fid			= chart_obj.fid
		# chart_data	= json.loads(chart_obj.chart_data)
		# layout		= json.loads(chart_obj.layout)
		
		# get fields
		table_data	= json.loads(chart_obj.table_data)
		fields = table_data[list(table_data.keys())[0]]['cols']
		sorted_fields = sorted(fields.items(), key=lambda x: x[1]['order'])
		num_fields = len(sorted_fields)
		if num_fields > MAX_FIELDS:
			fids_exceeding_max_fields.append(fid)
			continue

		# parsing fields
		parsed_fields = []
		for i, (field_name, field_data) in enumerate(sorted_fields):
			field_id		= field_data['uid']
			field_order		= field_data['order']
			field_values	= field_data['data']
			field_type, type_scores	= detect_field_type(field_values)
			field_general_type		= data_type_to_general_type[field_type]
			parsed_field = {
				'uid':			field_id,
				'name':			field_name,
				'order':		field_order,
				'data_type':	field_type,
				'general_type':	field_general_type,
			}
			try:
				v = parse(field_values, field_type, field_general_type)
				v = np.ma.array(v).compressed()
				parsed_field['data']		= v
				parsed_field['unique_data']	= get_unique(v)
			except Exception as e:
				extract_logger.log('[fid: {}] Error parsing {}'.format(fid, field_name))
				extract_logger.log('\t' + str(e))
			parsed_fields.append(parsed_field)

		# extract pairwise features
		pairwise_field_features = extract_pairwise_field_features(	# (25-1) * (24, 23, ..., 1) = 300
			parsed_fields,
			fid,
			extract_logger,
			MAX_FIELDS=MAX_FIELDS
		)

		pairwise_features_df = construct_chart_pairwise_features_df(pairwise_field_features)
		chunk_pairwise_features_df = pd.concat(
			[chunk_pairwise_features_df, pairwise_features_df],
			ignore_index=True
		)

		# flatten pairwise field features 暴力存储
		# flattened_pairwise_features = []		# 300 OrderedDict
		# for one_field_pairwise_field_features in pairwise_field_features:
		# 	flattened_pairwise_features.extend(one_field_pairwise_field_features)
		# pairwise_features_json = json.dumps(flattened_pairwise_features)
		# chunk_pairwise_features_df.append({'fid': fid, 'pairwise_features': pairwise_features_json})

	return chunk_pairwise_features_df



def write_batch_results(batch_results, first_batch):
	for df in batch_results:
		df.to_csv(
			pairwise_features_file,
			mode	= 'a',
			index	= False,
			header	= first_batch
		)



def main_process():
	extract_logger.log('Load raw data from:')
	extract_logger.log(raw_data_file)
	raw_df_chunks = pd.read_csv(
		raw_data_file,
		delimiter='\t',
		error_bad_lines=False,
		chunksize=chunk_size,
		encoding='utf-8'
	)
	extract_logger.log('')
	
	first_batch = True
	start_time = time.time()
	if parallelize:
		batch_size = multiprocessing.cpu_count()	# n_jobs
		extract_logger.log('Number of jobs: ' + str(batch_size))
		# start
		batch_num = 1
		chunk_batch = []
		for i, chunk in tqdm(enumerate(raw_df_chunks)):
			chunk_num = i + 1
			# chunk_batch.append({
			# 	'chunk':		chunk,
			# 	'batch_num':	batch_num,
			# 	'chunk_num': 	chunk_num
			# })
			chunk_batch.append(chunk)
			if chunk_num == (batch_size * batch_num):
				extract_logger.log('Start batch {} [chunk {} ~ {}]'.format(
								batch_num, chunk_num-batch_size+1, chunk_num))
				pool = multiprocessing.Pool(batch_size)
				batch_start_time = time.time()
				batch_results = pool.map_async(extract_in_a_chunk, chunk_batch).get(9999999)
				write_batch_results(batch_results, first_batch=first_batch)
				batch_time_cost = time.time() - batch_start_time
				extract_logger.log('Finish batch {}'.format(batch_num))
				extract_logger.log('Time cost: {:.1f}s'.format(batch_time_cost))
				extract_logger.log('')
				batch_num	+= 1
				chunk_batch	= []
				first_batch	= False
				pool.close()
		# process left overs
		if len(chunk_batch) != 0:
			extract_logger.log('Start last batch {} [chunk {} ~ {}]'.format(
							batch_num, batch_size*(batch_num-1)+1, chunk_num))
			pool = multiprocessing.Pool(batch_size)
			batch_start_time = time.time()
			remaining_batch_results = pool.map_async(extract_in_a_chunk, chunk_batch).get(9999999)
			write_batch_results(remaining_batch_results, first_batch=first_batch)
			batch_time_cost = time.time() - batch_start_time
			extract_logger.log('Finish last batch {}'.format(batch_num))
			extract_logger.log('Time cost: {:.1f}s'.format(batch_time_cost))
			extract_logger.log('')
			pool.close()

	# 不并行
	else:
		for i, chunk in tqdm(enumerate(raw_df_chunks)):
			chunk_num = i + 1
			extract_logger.log('Start chunk {}'.format(chunk_num))
			chunk_start_time = time.time()
			chunk_result = extract_in_a_chunk(chunk)
			write_batch_results([chunk_result], first_batch=first_batch)
			chunk_time_cost = time.time() - chunk_start_time
			extract_logger.log('Finish chunk {}'.format(chunk_num))
			extract_logger.log('Time cost: {:.1f}s'.format(chunk_time_cost))
			extract_logger.log('')
			first_batch	= False



if __name__ == '__main__':
	main_process()
