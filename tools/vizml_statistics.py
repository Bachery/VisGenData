from tqdm import tqdm
import pandas as pd
import numpy as np
import multiprocessing
import datetime
import pickle
# import copy # 效率太低
import time
import json
import csv
import sys

import IPython
from vizml_look_at_chart import *

sys.path.insert(0, '..')
from helpers.logger import logger
from feature_extraction.chart_outcomes import extract_chart_outcomes, outcome_properties


raw_data_file = '../data/vizml_ded/original_tsv/plot_data_with_all_fields_and_header_deduplicated_one_per_user.tsv'
chart_outcome_file = '../data/vizml_ded/features/chart_outcome.csv'
raw_chunks_folder = '../data/vizml_ded/raw_data/'

chunk_size = 1000
black_list = [
	'blasinzw:96',	# box，table_data里有两个字典，每个trace只有y轴，有一个trace出错复制多一个x轴，本来数据也差，结果图很难看
	'it_dcm:5325',	# scatter，但其实是line，table_data里有两个字典，其中一个是错把一个trace里的marker-size属性存作维度'text'，导致和原有的'text'名字冲突，而且这里画两个点意义不明
	'it_dcm:5330',	# 同上
	'it_dcm:5332',	# 同上
	'ybrumer:46',	# bar, 10201个trace，绘图报错，网站图空空
	'94harshsharma:7', # bar, chart里出现table里没有的uid
]


def check_layout_data(chunk_files, chart_type, statis_logger=None):
	# statistic outputs
	chart_num = 0
	empty_layout = 0
	layout_keys	= {}
	layout_axis	= {}
	axis_keys = {}
	for chunk_file in tqdm(chunk_files):
		chunk_df = pd.read_csv(raw_chunks_folder + chunk_file)
		for chart_idx, chart_obj in chunk_df.iterrows():
			fid = chart_obj.fid
			if fid in black_list: continue
			layout = json.loads(chart_obj.layout)
			chart_num += 1
			if len(layout.keys()) == 0:
				empty_layout += 1
				continue
			axis_names = []
			for k in layout.keys():
				kt = k + type(layout[k]).__name__
				if (not 'axis' in k) and (not kt in layout_keys.keys()): layout_keys[kt] = 1
				elif (not 'axis' in k) and (kt in layout_keys.keys()): layout_keys[kt] += 1
				elif ('axis' in k) and (not kt in layout_axis.keys()):
					layout_axis[kt] = 1
					axis_names.append(k)
					if k == 'xaxis2' or k == 'yaxis2':
						download_chart(fid, chart_type)
						draw_chart_plotly_json(chart_obj, chart_type)
					if isinstance(layout[k], dict):
						for kk in layout[k].keys():
							if not kk in axis_keys.keys(): axis_keys[kk] = 1
							else: axis_keys[kk] += 1
				else:
					layout_axis[kt] += 1
					if isinstance(layout[k], dict):
						for kk in layout[k].keys():
							if not kk in axis_keys.keys(): axis_keys[kk] = 1
							else: axis_keys[kk] += 1

	if statis_logger is not None:
		statis_logger.log('')
		statis_logger.log('-------------------------------------------------')
		statis_logger.log(chart_type)
		statis_logger.log('chart num: {}'.format(chart_num))
		statis_logger.log('empty layout: {}'.format(empty_layout))
		statis_logger.log('layout axis keys:')
		statis_logger.log_dict(layout_axis)
		statis_logger.log('')
		statis_logger.log('axis keys: ')
		statis_logger.log_dict(axis_keys)
		statis_logger.log('')
		statis_logger.log('other layout keys: ')
		statis_logger.log_dict(layout_keys)
		statis_logger.log('')



def check_chart_data(chunk_files, chart_type, statis_logger=None):
	# statistic outputs
	trace_num = []
	field_num = []
	trace_keys = {}
	src_keys = {}
	# have_a_look = ['opacity', 'marker', 'orientation', 'visible', 'mode', 'total'] # for bar
	marker_options = {}
	orientation_options = []
	mode_options = []
	# start traversing
	for chunk_file in tqdm(chunk_files):
		chunk_df = pd.read_csv(raw_chunks_folder + chunk_file)
		for chart_idx, chart_obj in chunk_df.iterrows():
			fid = chart_obj.fid
			if fid in black_list: continue
			user, idx	= fid.split(':')
			chart_data	= json.loads(chart_obj.chart_data)
			# layout		= json.loads(chart_obj.layout)
			# table_data	= json.loads(chart_obj.table_data)
			trace_num.append(len(chart_data))
			# 看看那些有超多trace的都是啥，结论：超10000的报错画不出来，超1000的都很花
			# if trace_num[-1] > 1000:
			# 	download_chart(fid, chart_type)
			# 	draw_chart_plotly_json(chart_obj, chart_type)
			for trace_id, trace in enumerate(chart_data):
				for trace_key in trace.keys():
					if 'src' in trace_key:
						if not trace_key in src_keys:
							# check各个src都怎么用的
							src_keys[trace_key] = 1
							# download_chart(fid, chart_type)
							# draw_chart_plotly_json(chart_obj, chart_type)
						else: src_keys[trace_key] += 1
					elif not trace_key in trace_keys: 
						trace_keys[trace_key] = 1
						# check 几个用得比较多的trace key是做什么的
						# if trace_key in have_a_look: 
						# 	download_chart(fid, chart_type)
						# 	draw_chart_plotly_json(chart_obj, chart_type)
					else: 
						trace_keys[trace_key] += 1
					
					if trace_key == 'orientation':
						if not trace[trace_key] in orientation_options:
							orientation_options.append(trace[trace_key])
							download_chart(fid, chart_type)
							draw_chart_plotly_json(chart_obj, chart_type)

				# for bar: 记录marker的key，orientation的选项，mode的选项都有哪些
					# if trace_key == 'marker':
					# 	if isinstance(trace['marker'], dict):
					# 		for marker_key in trace['marker'].keys():
					# 			if not marker_key in marker_options.keys():
					# 				if isinstance(trace['marker'], dict): 
					# 					marker_options[marker_key] = {}
					# 					for kk in trace['marker'].keys():
					# 						marker_options[marker_key][kk] = 1
					# 				else:
					# 					marker_options[marker_key] = 1
					# 			else:
					# 				if isinstance(trace['marker'], dict): 
					# 					for kk in trace['marker'].keys():
					# 						if not kk in marker_options[marker_key].keys():
					# 							marker_options[marker_key][kk] = 1
					# 						else:
					# 							marker_options[marker_key][kk] += 1
					# 				else:
					# 					marker_options[marker_key] += 1
					# 	# else: # 看一下有没有marker不是dict的，不是dict时是空列表[]，绘制会报错
					# 	# 	download_chart(fid, chart_type)
					# 	# 	draw_chart_plotly_json(chart_obj, chart_type)
					# if trace_key == 'orientation':
					# 	if not trace[trace_key] in orientation_options:
					# 		orientation_options.append(trace[trace_key])
					# if trace_key == 'mode':
					# 	if not trace[trace_key] in mode_options:
					# 		mode_options.append(trace[trace_key])
	
			
	nums, counts = np.unique(trace_num, return_counts=True)
	if statis_logger is not None:
		statis_logger.log('')
		statis_logger.log('-------------------------------------------------')
		statis_logger.log(chart_type)
		statis_logger.log('trace num(max/min/mean): %d/%d/%f ' % (max(trace_num), min(trace_num), np.mean(trace_num)))
		statis_logger.log('counts of traces nums:')
		statis_logger.log_dict(dict(zip(nums, counts)))
		statis_logger.log('')
		statis_logger.log('trace keys: ')
		statis_logger.log_dict(trace_keys)
		statis_logger.log('')
		statis_logger.log('src keys: ')
		statis_logger.log_dict(src_keys)
		statis_logger.log('')
	


def check_table_data(chunk_files, chart_type):
	# statistic outputs
	reasons = ['restricted']
	col_keys = ['uid', 'order', 'data']
	for chunk_file in tqdm(chunk_files):
		chunk_df = pd.read_csv(raw_chunks_folder + chunk_file)
		for chart_idx, chart_obj in chunk_df.iterrows():
			fid = chart_obj.fid
			if fid in black_list: continue
			table_data	= json.loads(chart_obj.table_data)
			# 验证table_data下是不是都只有一个字典，结论：不是，可能有多个
			if len(table_data.keys()) != 1:
				download_chart(fid, chart_type)
				draw_chart_plotly_json(chart_obj, chart_type)
			table_data_1 = table_data[list(table_data.keys())[0]]
			# 验证table_data里的字典只有reason和cols两项，结论：是的
			if len(table_data_1.keys()) != 2:
				download_chart(fid, chart_type)
				draw_chart_plotly_json(chart_obj, chart_type)
				IPython.embed()
			# 验证table_data里字典都有reason项，且值只有‘restricted’，结论：是的
			if not 'reason' in table_data_1.keys():
				download_chart(fid, chart_type)
				draw_chart_plotly_json(chart_obj, chart_type)
				IPython.embed()
			if not table_data_1['reason'] in reasons:
				download_chart(fid, chart_type)
				draw_chart_plotly_json(chart_obj, chart_type)
				print(table_data_1['reason'])
				reasons.append(table_data_1['reason'])
			# fields = table_data[list(table_data.keys())[0]]['cols']
			# 出现了有多个字典存着多组数据的情况，直接取0会找不到另外组里的
			# 不同组的维度的order互相无关
			# 有时甚至，多组字典的某条数据内容相同uid却不同
			# 验证uid不同但名称相同的维度数据是否都相同，结论：名称相同则数据相同，可合并
			fields_by_name = {}
			table_data_dict_num = len(table_data.keys())
			for dict_id in range(table_data_dict_num):	# 不同组的fields有可能名称相同，update不适用
				# fields_by_name.update(table_data[list(table_data.keys())[dict_id]]['cols'])
				new_fields = table_data[list(table_data.keys())[dict_id]]['cols']
				for new_name in new_fields.keys():
					if new_name in fields_by_name.keys():
						if fields_by_name[new_name]['data'] != new_fields[new_name]['data']:
							IPython.embed()
				fields_by_name.update(new_fields)#(copy.deepcopy(new_fields))
			# 根据id存field就不会名字冲突，但是有可能忽略某些field是一样的
			fields_by_id = {}
			table_data_dict_num = len(table_data.keys())
			for dict_id in range(table_data_dict_num):
				fields = table_data[list(table_data.keys())[dict_id]]['cols']	
				for field_name, d in fields.items():
					# 验证fields的内容都有哪些，结论：uid order data都有，name有的有
					# 研究下内部的name
					for field_key in d.keys():
						if not field_key in col_keys:
							download_chart(fid, chart_type)
							draw_chart_plotly_json(chart_obj, chart_type)
							IPython.embed()
					# 验证不同组的field的uid是否有可能重复，结论：不会
					if d['uid'] in fields_by_id.keys():
						IPython.embed()
					fields_by_id[d['uid']] = d#copy.deepcopy(d)
					fields_by_id[d['uid']]['field_name'] = field_name



def statistics_one_type(chart_type, statis_logger=None):
	def list_dir(folder):
		contents = os.listdir(folder)
		if '.DS_Store' in contents: 
			contents.remove('.DS_Store')
		return sorted(contents)
	# load outcomes
	# statis_logger.log('Load chart outcomes from:')
	# statis_logger.log(chart_outcome_file)
	# chart_outcome_df = pd.read_csv(chart_outcome_file)
	# chart_outcome_df = pd.DataFrame(chart_outcome_df).set_index('fid') # set 'fid' as index for outcomes df
	# load chunk data
	chunk_files = list_dir(raw_chunks_folder)
	chunk_files = sorted([f for f in chunk_files if chart_type in f])
	# chunk_files = chunk_files[11:]
	# check_table_data(chunk_files, chart_type)
	check_layout_data(chunk_files, chart_type, statis_logger)
	# check_chart_data(chunk_files, chart_type, statis_logger)


def save_raw_by_types():
	log_suffix = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
	log_file = 'logs/' + 'vizml_ded_partition_{}.txt'.format(log_suffix)
	statis_logger = logger(log_file, {'Mission': 'seperate charts by types for ded charts'})
	# load outcomes
	statis_logger.log('Load chart outcomes from:')
	statis_logger.log(chart_outcome_file)
	chart_outcome_df = pd.read_csv(chart_outcome_file)
	chart_outcome_df = pd.DataFrame(chart_outcome_df).set_index('fid') # set 'fid' as index for outcomes df
	# load raw data
	statis_logger.log('Load raw data from:')
	statis_logger.log(raw_data_file)
	raw_df_chunks = pd.read_csv(
		raw_data_file,
		delimiter='\t',
		error_bad_lines=False,
		chunksize=chunk_size,
		encoding='utf-8'
	)
	statis_logger.log('')
	# chunks_by_types
	cols = ['fid', 'chart_data', 'layout', 'table_data']
	bar_chunk		= pd.DataFrame(columns=cols)
	box_chunk		= pd.DataFrame(columns=cols)
	heatmap_chunk	= pd.DataFrame(columns=cols)
	histogram_chunk	= pd.DataFrame(columns=cols)
	line_chunk		= pd.DataFrame(columns=cols)
	pie_chunk		= pd.DataFrame(columns=cols)
	scatter_chunk	= pd.DataFrame(columns=cols)
	chunks_nums		= [1] * 7
	# statistics
	seen_fids			= []
	num_duplicates		= 0
	num_no_outcome		= 0
	num_multi_subplots	= 0
	num_multi_types		= 0
	num_other_types		= 0

	def save_type_chunk(chunk_type, saving_chunk, chunk_idx):
		save_file = raw_chunks_folder + '{}_chunk_{}.csv'.format(chunk_type, chunk_idx)
		saving_chunk.to_csv(save_file, index=False)
		statis_logger.log('Saving: {}'.format(save_file))
		return chunk_idx+1, pd.DataFrame(columns=cols)

	# traverse raw data
	for i, chunk in tqdm(enumerate(raw_df_chunks)):
		chunk_num = i + 1
		statis_logger.log('Start chunk {}'.format(chunk_num))
		chunk_start_time = time.time()
		for chart_idx, chart_obj in chunk.iterrows():
			fid = chart_obj.fid
			# skip duplicated fid (only keep the first one)
			if fid in seen_fids:
				num_duplicates += 1
				continue
			else:
				seen_fids.append(fid)
			if not fid in chart_outcome_df.index:
				num_no_outcome += 1
				continue
			# skip multi-subplots or multi-types
			if_one_subplot		= chart_outcome_df.loc[fid, 'one_subplot']
			if_one_trace_type	= chart_outcome_df.loc[fid, 'is_all_one_trace_type']
			if not if_one_subplot:
				num_multi_subplots += 1
				continue
			if not if_one_trace_type:
				num_multi_types += 1
				continue
			chart_type = chart_outcome_df.loc[fid, 'all_one_trace_type']
			if   chart_type == 'bar':		bar_chunk		= bar_chunk.append(			[chart_obj], ignore_index=True)
			elif chart_type == 'box':		box_chunk		= box_chunk.append(			[chart_obj], ignore_index=True)
			elif chart_type == 'heatmap':	heatmap_chunk	= heatmap_chunk.append(		[chart_obj], ignore_index=True)
			elif chart_type == 'histogram':	histogram_chunk	= histogram_chunk.append(	[chart_obj], ignore_index=True)
			elif chart_type == 'line':		line_chunk		= line_chunk.append(		[chart_obj], ignore_index=True)
			elif chart_type == 'pie':		pie_chunk		= pie_chunk.append(			[chart_obj], ignore_index=True)
			elif chart_type == 'scatter':	scatter_chunk	= scatter_chunk.append(		[chart_obj], ignore_index=True)
			else: num_other_types += 1
			# save to file if reach chunk size
			if bar_chunk.shape[0]			== chunk_size:
				chunks_nums[0], bar_chunk		= save_type_chunk('bar',		bar_chunk,			chunks_nums[0])
			elif box_chunk.shape[0]			== chunk_size:
				chunks_nums[1], box_chunk		= save_type_chunk('box',		box_chunk,			chunks_nums[1])
			elif heatmap_chunk.shape[0]		== chunk_size:
				chunks_nums[2], heatmap_chunk	= save_type_chunk('heatmap',	heatmap_chunk,		chunks_nums[2])
			elif histogram_chunk.shape[0]	== chunk_size:
				chunks_nums[3], histogram_chunk	= save_type_chunk('histogram',	histogram_chunk,	chunks_nums[3])
			elif line_chunk.shape[0]		== chunk_size:
				chunks_nums[4], line_chunk		= save_type_chunk('line',		line_chunk,			chunks_nums[4])
			elif pie_chunk.shape[0]			== chunk_size:
				chunks_nums[5], pie_chunk		= save_type_chunk('pie',		pie_chunk,			chunks_nums[5])
			elif scatter_chunk.shape[0]		== chunk_size:
				chunks_nums[6], scatter_chunk	= save_type_chunk('scatter',	scatter_chunk,		chunks_nums[6])
		# finish each chunk
		chunk_time_cost = time.time() - chunk_start_time
		statis_logger.log('Finish chunk {}'.format(chunk_num))
		statis_logger.log('Time cost: {:.1f}s'.format(chunk_time_cost))
		statis_logger.log('')
	# finish all, save the rest
	if bar_chunk.shape[0]		> 0: save_type_chunk('bar',			bar_chunk,			chunks_nums[0])
	if box_chunk.shape[0]		> 0: save_type_chunk('box',			box_chunk,			chunks_nums[1])
	if heatmap_chunk.shape[0]	> 0: save_type_chunk('heatmap',		heatmap_chunk,		chunks_nums[2])
	if histogram_chunk.shape[0]	> 0: save_type_chunk('histogram',	histogram_chunk,	chunks_nums[3])
	if line_chunk.shape[0]		> 0: save_type_chunk('line',		line_chunk,			chunks_nums[4])
	if pie_chunk.shape[0]		> 0: save_type_chunk('pie',			pie_chunk,			chunks_nums[5])
	if scatter_chunk.shape[0]	> 0: save_type_chunk('scatter',		scatter_chunk,		chunks_nums[6])
	# log statistics
	statis_logger.log('')
	statis_logger.log('      bar num: {}'.format(chunks_nums[0] * chunk_size + bar_chunk.shape[0]))
	statis_logger.log('      box num: {}'.format(chunks_nums[1] * chunk_size + box_chunk.shape[0]))
	statis_logger.log('  heatmap num: {}'.format(chunks_nums[2] * chunk_size + heatmap_chunk.shape[0]))
	statis_logger.log('histogram num: {}'.format(chunks_nums[3] * chunk_size + histogram_chunk.shape[0]))
	statis_logger.log('     line num: {}'.format(chunks_nums[4] * chunk_size + line_chunk.shape[0]))
	statis_logger.log('      pie num: {}'.format(chunks_nums[5] * chunk_size + pie_chunk.shape[0]))
	statis_logger.log('  scatter num: {}'.format(chunks_nums[6] * chunk_size + scatter_chunk.shape[0]))
	statis_logger.log('')
	statis_logger.log('unique fid num: {}'.format(len(seen_fids)))
	statis_logger.log('duplicated num: {}'.format(num_duplicates))
	statis_logger.log('no outcome num: {}'.format(num_no_outcome))
	statis_logger.log('multi-subplots num: {}'.format(num_multi_subplots))
	statis_logger.log('multi-types num: {}'.format(num_multi_types))
	statis_logger.log('other-types num: {}'.format(num_other_types))


def read_from_outcome_files():
	statis_logger.log('Load chart outcomes from:')
	statis_logger.log(chart_outcome_file)
	chart_outcome_df = pd.read_csv(chart_outcome_file)
	# print(chart_outcome_df.shape)
	# print(chart_outcome_df.columns)
	return chart_outcome_df


def deduplicate_outcome_file(file):
	# 处理 main_process 保存的outcome文件，去除fid相同的行
	df = pd.read_csv(file)
	print(df.shape)
	df = df.drop_duplicates('fid', keep='first')
	print(df.shape)
	df.to_csv(file, index=False)


def extract_chart_outcomes_from_ded_tsv():
	log_suffix = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
	log_file = 'logs/' + 'vizml_ded_statistics_{}.txt'.format(log_suffix)
	statis_logger = logger(log_file, {'Mission': 'statistics for ded charts'})
	statis_logger.log('Load raw data from:')
	statis_logger.log(raw_data_file)
	raw_df_chunks = pd.read_csv(
		raw_data_file,
		delimiter='\t',
		error_bad_lines=False,
		chunksize=chunk_size,
		encoding='utf-8'
	)
	statis_logger.log('')

	num_fields = []
	num_traces = []
	trace_types = []
	if_one_subplot = []
	if_one_trace_type = []
	one_trace_type = []
	if_single_src = []

	first_batch = True
	outcome_df = pd.DataFrame(columns=outcome_properties)

	for i, chunk in tqdm(enumerate(raw_df_chunks)):
		chunk_num = i + 1
		statis_logger.log('Start chunk {}'.format(chunk_num))
		chunk_start_time = time.time()
		for chart_idx, chart_obj in chunk.iterrows():

			chart_outcomes = None
			try:
				chart_outcomes = extract_chart_outcomes(chart_obj)
			except Exception as e:
				statis_logger.log('[{}] Uncaught exception'.format(chart_obj.fid))
				statis_logger.log('\t' + str(e))

			if not chart_outcomes is None:
				num_fields.append(			chart_outcomes['num_fields_used_by_data'])
				num_traces.append(			chart_outcomes['num_traces'])
				trace_types.extend(			chart_outcomes['trace_types'])
				if_one_subplot.append(		chart_outcomes['one_subplot'])
				if_one_trace_type.append(	chart_outcomes['is_all_one_trace_type'])
				if chart_outcomes['is_all_one_trace_type']:
					one_trace_type.append(	chart_outcomes['all_one_trace_type'])
				if_single_src.append(		chart_outcomes['has_single_src'])

				outcome_df = outcome_df.append([chart_outcomes], ignore_index=True)
		
		outcome_df.to_csv(chart_outcome_file, mode='a', index=False, header=first_batch)
		first_batch = False
		outcome_df = pd.DataFrame(columns=outcome_properties)

		chunk_time_cost = time.time() - chunk_start_time
		statis_logger.log('Finish chunk {}'.format(chunk_num))
		statis_logger.log('Time cost: {:.1f}s'.format(chunk_time_cost))
		statis_logger.log('')


	statis_logger.log(	'chart num: %d'			%	len(num_traces)	)
	statis_logger.log(	'avg fields num: %d'	%	np.mean(num_fields)	)
	statis_logger.log(	'avg traces num: %d'	%	np.mean(num_traces)	)
	statis_logger.log(	'one subplot: %d'		%	np.sum(if_one_subplot)	)
	statis_logger.log(	'one trace type: %d'	%	np.sum(if_one_trace_type)	)
	statis_logger.log(	'has single src: %d'	%	np.sum(if_single_src)	)
	statis_logger.log('')

	unique, counts = np.unique(trace_types, return_counts=True)
	statis_logger.log('Value counts of traces types:')
	statis_logger.log_dict(dict(zip(unique, counts)))
	statis_logger.log('')

	unique2, counts2 = np.unique(one_trace_type, return_counts=True)
	statis_logger.log('Value counts of one-trace types:')
	statis_logger.log_dict(dict(zip(unique2, counts2)))
	statis_logger.log('')

	
	statis_logger.log('save chart outcomes to: ')
	statis_logger.log(chart_outcome_file)



if __name__ == '__main__':
	# log setting
	log_suffix = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
	log_file = 'logs/' + 'vizml_ded_statistics_layout_{}.txt'.format(log_suffix)
	# statis_logger = logger(log_file, {'Mission': 'layout statistics for ded charts'})

	# extract_chart_outcomes_from_ded_tsv()
	# deduplicate_outcome_file(chart_outcome_file)
	# read_from_outcome_files()
	# save_raw_by_types()

	# types = ['bar', 'box', 'heatmap', 'histogram', 'line', 'pie', 'scatter']
	# for t in types:
	# 	statistics_one_type(t, statis_logger)

	statistics_one_type('bar')