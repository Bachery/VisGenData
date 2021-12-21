
from tqdm import tqdm
import pandas as pd
import numpy as np
import datetime
# import pickle
# import copy
# import time
import json
# import csv
import sys
import os

import IPython
from vizml_look_at_chart import *
from vizml_statistics import black_list

sys.path.insert(0, '..')
from helpers.logger import logger
from helpers.utils import list_dir
from feature_extraction.type_detection import detect_field_type, data_type_to_general_type
from Modules.insight import Insight

dataset = 'vizml_ded'
raw_chunks_folder	= '../data/{}/raw_data/'.format(dataset)
insight_save_folder	= '../data/{}/insights/'.format(dataset)
chunk_size = 1000

black_list.extend([
	'aust1n:3287'		# too large, raise MemoryError
])
black_list_user = [
	'automata'			# scatter but multi-lines, has more than 20,000 figures
]

# TODO 加入chart_evaluator里对各个chart_type的基础filtering，只留下满足保底条件的图表


def format_data_to_series(data, field_type, fid='', trans_logger=None):
	"""
	Input:
		data:		array-like
		field_type:	str [integer / decimal / time / string]
	Output:
		series:		pandas Series with dtypes [Int64Dtype / float64 / datetime / object]
	"""
	assert field_type in ['integer', 'decimal', 'time', 'string']
	if len(data) == 0:
		if trans_logger: trans_logger.log('[{}] has data length == 0'.format(fid))
		return pd.Series([])
	show_length = 5 if len(data)>=5 else len(data)	# 记录错误信息时展示部分数据

	if field_type == 'integer':
		try:
			return pd.Series(data, dtype=pd.Int64Dtype())
		except Exception as e:
			result = []
			for value in data:					# 直接转换有异常，尝试逐个数据转换，记录到列表
				try:
					result.append(int(value))
				except ValueError as ve:		# 个别异常数值填充nan，避免维度长度变化
					result.append(np.nan)
				except Exception as e:			# 记录错误，数据类型转为Categorical
					if trans_logger:
						trans_logger.log('[{}] exception formatting data to Int64Dtype: {}'.format(fid, value))
						trans_logger.log('\t {}'.format(e))
					return pd.Series(data)
			try:
				result = pd.Series(result, dtype=pd.Int64Dtype())	# 指定dtype保留nan
				if result.dropna().shape[0] == 0:	# 全是NaN，记录错误，数据类型转为Categorical
					trans_logger.log('[{}] integer all transform as NaN: {}'.format(fid, data[:show_length]))
					return pd.Series(data)
			except Exception as e:		# 逐个int了还是无法转为Int64Dype，记录错误，数据类型转为Categorical
				if trans_logger: 
					trans_logger.log('[{}] exception formatting data to Int64Dtype: {}'.format(fid, data[:show_length]))
					trans_logger.log('\t {}'.format(e))
				return pd.Series(result)	# 转为 C type
			return result

	elif field_type == 'decimal':
		try:
			return pd.Series(data, dtype='float64')
		except Exception as e:
			result = []
			for value in data:					# 直接转换有异常，尝试逐个数据转换，记录到列表
				try:
					result.append(float(value))
				except ValueError as ve:		# 个别异常数值填充nan，避免维度长度变化
					result.append(np.nan)
				except Exception as e:			# 记录错误，数据类型转为Categorical
					if trans_logger:
						trans_logger.log('[{}] exception formatting data to float64: {}'.format(fid, value))
						trans_logger.log('\t {}'.format(e))
					return pd.Series(data)
			try:
				result = pd.Series(result, dtype='float64')		# 指定dtype保留nan
				if result.dropna().shape[0] == 0:	# 全是NaN，记录错误，数据类型转为Categorical
					trans_logger.log('[{}] decimal all transform as NaN: {}'.format(fid, data[:show_length]))
					return pd.Series(data)
			except Exception as e:		# 逐个float了还是无法转为float64，记录错误，数据类型转为Categorical
				if trans_logger:
					trans_logger.log('[{}] exception formatting data: {}'.format(fid, data[:show_length]))
					trans_logger.log('\t {}'.format(e))
				return pd.Series(result)
			return result

	elif field_type == 'time':
		try:
			result = pd.Series(pd.to_datetime(
				data,
				errors='coerce',		# 无法解析的格式填充NaT
				infer_datetime_format=True,
				utc=True				# UTC格式，不受时区影响的时间
			))
			if result.dropna().shape[0] == 0:	# 全是NaT，记录错误，数据类型转为Categorical
				trans_logger.log('[{}] datetime all transform as NaT: {}'.format(fid, data[:show_length]))
				return pd.Series(data)
		except Exception as e:
			if trans_logger:
				trans_logger.log('[{}] datetime transform fail: {}'.format(fid, data[:show_length]))
				trans_logger.log('\t {}'.format(e))
			return pd.Series(data)
		return result

	else:
		return pd.Series(data)
	

def extract_table_fields(fid, table_data, trans_logger=None):
	# table_data可能有多个字典存着多组数据的情况
	# 不同组的维度的order互相无关
	# 有时，不同组间有uid不同但data和name却相同的数据
	# 已验证：名称相同时data也相同，可合并
	fields_by_name = {}
	uid_name_mapping = {}
	uids_by_dict = []	# sorted by order
	table_data_dict_num = len(table_data.keys())
	for dict_id in range(table_data_dict_num):	# 不同组的fields有可能名称相同，update不适用
		# fields_by_name.update(table_data[list(table_data.keys())[dict_id]]['cols'])
		new_fields = table_data[list(table_data.keys())[dict_id]]['cols']
		fields_by_name.update(new_fields)
		sorted_fields = sorted(new_fields.items(), key=lambda x: x[1]['order'])
		sorted_new_uids  = [f[1]['uid']	for f in sorted_fields]
		sorted_new_names = [f[0]		for f in sorted_fields]
		uid_name_mapping.update( dict(zip(sorted_new_uids, sorted_new_names)) )
		uids_by_dict.append(sorted_new_uids)
		
	for field_name, field in fields_by_name.items():
		field_type, _ = detect_field_type(field['data'])			# string, integer, decimal, time
		field_general_type = data_type_to_general_type[field_type]	# c, q, t
		field['type'] = field_general_type.upper()					# C, Q, T
		field['name'] = field_name
		# format date to pd.Series
		field['data'] = format_data_to_series(field['data'], field_type, trans_logger)
		if (field_type == 'time') and (field['data'].dtype == 'object'):
			field['type'] = 'C'		# fail to transform to datetime
		if (field_type == 'integer') and (field['data'].dtype == 'object'):
			field['type'] = 'C'		# fail to transform to int
		if (field_type == 'decimal') and (field['data'].dtype == 'object'):
			field['type'] = 'C'		# fail to transform to float
	return fields_by_name, uid_name_mapping, uids_by_dict


def get_field_from_src(fid, src, fields_by_name, uid_name_mapping, uids_by_dict, trans_logger=None):
	uid = src.split(':')[-1]
	# 根据uid取数据

	if len(uid) == 0:	# 根据heatmap的情况看，没有uid时使用所有的src
		# if trans_logger: trans_logger.log('[{}]({}) error uid format'.format(fid, src))
		# return None
		using_uids = [uuid for dict_uids in uids_by_dict for uuid in dict_uids]
		uid = ','.join(using_uids)
	
	# 开头为'-'，代表的是，除了这些uid外的全部uid，应该来自table_data的同一个dict，根据order排序
	if uid.startswith('-'):
		abandon_uids = uid[1:].split(',')
		using_dict = []
		for dict_uids in uids_by_dict:
			if abandon_uids[0] in dict_uids:
				using_dict = dict_uids
				break
		if len(using_dict) == 0:		# 有一个dict包含这个带'-'的uid
			if trans_logger: trans_logger.log('[{}]({}) -uid not found'.format(fid, uid))
			return None
		for abandon_uid in abandon_uids:
			assert abandon_uid in using_dict	# 所有被排除的uid应该在同一个dict里
		using_uids = [uuid for uuid in using_dict if not uuid in abandon_uids]	# sorted
		uid = ','.join(using_uids)
		# used_names = [uid_name_mapping[uuid] for uuid in used_uids]
	
	# 一个uid
	if len(uid) == 6:
		if not uid in uid_name_mapping.keys():
			if trans_logger: trans_logger.log('[{}]({}) uid not found'.format(fid, uid))
			return None
		field = fields_by_name[ uid_name_mapping[ uid ] ]

	# 一长串uid
	else:
		uid_list = uid.split(',')
		names = []
		types = []
		datas = []
		for uuid in uid_list:
			if not uuid in uid_name_mapping.keys():
				if trans_logger: trans_logger.log('[{}]({}) uuid not found'.format(fid, uuid))
				return None
			ff = fields_by_name[ uid_name_mapping[ uuid ] ]
			names.append(uid_name_mapping[ uuid ])
			types.append(ff['type'])
			datas.append(ff['data'])	# 每个data是一个pd.Series
		if len(set(types)) != 1:
			if trans_logger: trans_logger.log('[{}]({}) field types not consistent'.format(fid, src))
			return None
		field = {}
		field['name'] = names[0][0]
		field['type'] = types[0]
		field['data'] = pd.concat(datas, ignore_index=True)	# 把矩阵按列拼成一个长列表
		field['matrix'] = pd.DataFrame( dict( zip(names, datas) ) )
	
	# 最后check数据还有没问题
	if len(field['data']) == 0:
		if trans_logger: trans_logger.log('[{}]({}) data length == 0'.format(fid, src))
		return None

	return field

"""
def transform_bar_chart_data_old(chart_obj, fields_by_name, uid_name_mapping, uids_by_dict, trans_logger=None):
	# init
	chart_type	= 'bar'
	fid			= chart_obj.fid
	chart_data	= json.loads(chart_obj.chart_data)
	layout		= json.loads(chart_obj.layout)
	insight = Insight()
	insight.vizml_fid	= fid
	insight.chart_type	= chart_type
	# layout extraction
	x_title, y_title = '', ''
	if 'title'   in layout.keys(): insight.title = layout['title']
	if 'barmode' in layout.keys() and isinstance(layout['barmode'], str): 
		insight.chart_descriptions['barmode'] = layout['barmode'] # stack, group, overlay, ...
	if 'xaxis'   in layout.keys() and 'title' in layout['xaxis']: x_title = layout['xaxis']['title']
	if 'yaxis'   in layout.keys() and 'title' in layout['yaxis']: y_title = layout['yaxis']['title']

	trace_num = len(chart_data)
	if trace_num > 40: 
		if trans_logger: trans_logger.log('[{}] too many traces'.format(fid))
		return None
	
	# 单个trace
	if trace_num == 1:
		trace = chart_data[0]
		# get orientation
		orientation = 'vertical' # none or 'v', 'l', 'vertical'
		if 'orientation' in trace.keys(): 
			if trace['orientation'] == 'h':
				orientation = 'horizontal'
		insight.chart_descriptions['orientation'] = orientation
		if 'opacity' in trace.keys():
			insight.m_chan_descriptions.append({'opacity': trace['opacity']})
		# get x and y
		xsrc, ysrc = trace.get('xsrc'), trace.get('ysrc')
		x_field, y_field = None, None
		if xsrc: x_field = get_field_from_src(fid, xsrc, fields_by_name, uid_name_mapping, uids_by_dict, trans_logger)
		if ysrc: y_field = get_field_from_src(fid, ysrc, fields_by_name, uid_name_mapping, uids_by_dict, trans_logger)
		if x_field and x_title and isinstance(x_title, str): x_field['name'] = x_title	# 使用layout里的title作为维度名称
		if y_field and y_title and isinstance(y_title, str): y_field['name'] = y_title	# 有时title是一个列表。。无语
		if x_field==None and y_field==None:		# x和y都没有
			if trans_logger: trans_logger.log('[{}] [single-trace] neither xsrc or ysrc'.format(fid))
			return None
		elif x_field==None or y_field==None:	# 只有一个
			if (y_field==None and orientation=='vertical') or (x_field==None and orientation=='horizontal'):
				if trans_logger: trans_logger.log('[{}] [single-trace] single channel with wrong orientation'.format(fid))
				return None
			if orientation=='vertical':
				insight.breakdown_channels = ['x']
				insight.measure_channels = ['y']
				field = y_field
			else:
				insight.breakdown_channels = ['y']
				insight.measure_channels = ['x']
				field = x_field
			insight.transformed_table	= pd.DataFrame({field['name']: field['data']})
			insight.breakdown_fields	= ['*']
			insight.breakdown_types		= ['*']
			insight.measure_fields		= [field['name']]
			insight.measure_types		= [field['type']]
		else:	# xsrc、ysrc都有
			# match length
			if len(x_field['data']) != len(y_field['data']):
				length = min(len(x_field['data']), len(y_field['data']))
				x_field['data'] = x_field['data'][:length]
				y_field['data'] = y_field['data'][:length]
			if orientation == 'vertical':		# 竖直bar的breakdown在x轴
				breakdown, measure = x_field, y_field
				insight.breakdown_channels	= ['x']
				insight.measure_channels	= ['y']
			elif orientation == 'horizontal':	# 水平bar的breakdown在y轴
				breakdown, measure = y_field, x_field
				insight.breakdown_channels	= ['y']
				insight.measure_channels	= ['x']
			insight.transformed_table	= pd.DataFrame({
				breakdown['name']:	breakdown['data'],
				measure['name']:	measure['data'],
			})
			insight.breakdown_fields	= [breakdown['name']]
			insight.breakdown_types		= [breakdown['type']]
			insight.measure_fields		= [measure['name']]
			insight.measure_types		= [measure['type']]
	
	# 有多个trace
	else:
		# 没有指定barmode是plotly默认是group
		# if not 'barmode' in layout.keys():
		# 	download_chart(fid, 'bar')
		# 	draw_chart_plotly_json(chart_obj, 'bar')
		if x_title=='' or not isinstance(x_title, str): x_title = 'x'
		if y_title=='' or not isinstance(y_title, str): y_title = 'y'
		x_fields = []
		y_fields = []
		orientations = ['vertical'] * trace_num
		trace_names = []
		opacity = []
		for trace_id, trace in enumerate(chart_data):
			xsrc, ysrc = trace.get('xsrc'), trace.get('ysrc')
			x_field, y_field = None, None
			ori  = trace.get('orientation')
			name = trace.get('name')
			opa  = trace.get('opacity')
			if xsrc: 
				x_field = get_field_from_src(fid, xsrc, fields_by_name, uid_name_mapping, uids_by_dict, trans_logger)
				if x_field is not None: x_fields.append(x_field)
			if ysrc: 
				y_field = get_field_from_src(fid, ysrc, fields_by_name, uid_name_mapping, uids_by_dict, trans_logger)
				if y_field is not None: y_fields.append(y_field)
			if x_field is not None and y_field is not None:
				# match length
				if len(x_field['data']) != len(y_field['data']):
					length = min(len(x_field['data']), len(y_field['data']))
					x_field['data'] = x_field['data'][:length]
					y_field['data'] = y_field['data'][:length]
			if ori and ori=='h':  orientations[trace_id] = 'horizontal'
			if name: 
				if not isinstance(name, list): trace_names.append(str(name))
				else: trace_names.append(str(name[0]))
			# else: trace_names.append('trace_{}'.format(trace_id))		# 保证name列表和field列表等长
			else: trace_names.append('')
			if opa:  opacity.append(opa)
		if len(set(orientations)) > 1:
			if trans_logger: trans_logger.log('[{}] [multi-trace] corss-trace orientations not match'.format(fid))
			return None
		insight.chart_descriptions['orientation'] = orientations[0]
		x_num, y_num = len(x_fields), len(y_fields)
		if x_num==0 and y_num==0:	# x和y都没有
			if trans_logger: trans_logger.log('[{}] [multi-trace] neither xsrc or ysrc'.format(fid))
			return None
		elif x_num==0 or y_num==0:	# 只有一个
			if (y_num==0 and orientations[0]=='vertical') or (x_num==0 and orientations[0]=='horizontal'):
				if trans_logger: trans_logger.log('[{}] [multi-trace] single channel with wrong orientation'.format(fid))
				return None
			if orientations[0]=='vertical':
				insight.breakdown_channels = ['x', 'color']
				insight.measure_channels = ['y']
				measure_title = y_title
				fields = y_fields
			else:
				insight.breakdown_channels = ['y', 'color']
				insight.measure_channels = ['x']
				measure_title = x_title
				fields = x_fields
			types, names, measure_data = [], [], []
			length = len(fields[0]['data'])
			for trace_id, f in enumerate(fields):
				if len(f['data']) != length:
					if trans_logger: trans_logger.log('[{}] [multi-trace] single channel with unmatch lengths'.format(fid))
					return None
				types.append(f['type'])
				if len(trace_names) == len(fields):	names.extend([trace_names[trace_id]] * length)
				else: 								names.extend([f['name']] * length)
				measure_data.extend(f['data'])
			if len(set(types)) != 1:
				if trans_logger: trans_logger.log('[{}] [multi-trace] field types not consistent'.format(fid))
				return None
			insight.transformed_table	= pd.DataFrame({
				'color':		pd.Series(names),
				measure_title:	pd.Series(measure_data),
			})
			insight.breakdown_fields	= ['*', 'color']
			insight.breakdown_types		= ['*', 'C']
			insight.measure_fields		= [measure_title]
			insight.measure_types		= [types[0]]
		else:	# x和y都有
			if x_num != y_num:
				if trans_logger: trans_logger.log('[{}] [multi-trace] x-y trace num not match'.format(fid))
				return None
			if orientations[0] == 'vertical':
				breakdowns, measures = x_fields, y_fields
				insight.breakdown_channels	= ['x', 'color']
				insight.measure_channels	= ['y']
				breakdown_title, measure_title = x_title, y_title
			else:
				breakdowns, measures = y_fields, x_fields
				insight.breakdown_channels	= ['y', 'color']
				insight.measure_channels	= ['x']
				breakdown_title, measure_title = y_title, x_title
			color_data, breakdown_data, measure_data = [], [], []
			breakdown_names, breakdown_types, measure_types = [], [], []
			for trace_id in range(x_num):
				length = len(measures[trace_id]['data']) # 前面已经检查过同个trace内长度相同
				if len(trace_names) == x_num:	color_data.extend([trace_names[trace_id]] * length)
				else: 							color_data.extend([measures[trace_id]['name']] * length)
				breakdown_data.extend(breakdowns[trace_id]['data'])
				breakdown_names.append(breakdowns[trace_id]['name'])
				breakdown_types.append(breakdowns[trace_id]['type'])
				measure_data.extend(measures[trace_id]['data'])
				measure_types.append(measures[trace_id]['type'])
			if len(set(breakdown_types)) != 1:
				if trans_logger: trans_logger.log('[{}] [multi-trace] breakdown types not consistent'.format(fid))
				return None
			if len(set(measure_types)) != 1:
				if trans_logger: trans_logger.log('[{}] [multi-trace] measure types not consistent'.format(fid))
				return None
			if len(set(breakdown_names)) == 1:
				breakdown_title = breakdown_names[0]
			# construct insight
			insight.transformed_table	= pd.DataFrame({
				breakdown_title:	pd.Series(breakdown_data),
				'color':			pd.Series(color_data),
				measure_title:		pd.Series(measure_data),
			})
			insight.breakdown_fields	= [breakdown_title, 'color']
			insight.breakdown_types		= [breakdown_types[0], 'C']
			insight.measure_fields		= [measure_title]
			insight.measure_types		= [measure_types[0]]

	
	insight.transformed_table = insight.transformed_table.dropna()
	if insight.transformed_table.shape[0] == 0:
		if trans_logger: trans_logger.log('[{}] all NaN Error'.format(fid))
		return None	
	return insight
"""

def transform_bar_chart_data(chart_obj, fields_by_name, uid_name_mapping, uids_by_dict, trans_logger=None):
	# init
	chart_type	= 'bar'
	fid			= chart_obj.fid
	if fid == 'Umbric:106':
		print()
	chart_data	= json.loads(chart_obj.chart_data)
	layout		= json.loads(chart_obj.layout)
	insight = Insight()
	insight.vizml_fid	= fid
	insight.chart_type	= chart_type
	abandon_srcs = ['tsrc', 'rsrc', 'zsrc', 'offsetsrc', 'basesrc']	# 含有这些src的图表都不要（trace数量分别为307, 305, 4, 2, 11）

	# layout extraction
	x_title, y_title = '', ''
	if 'title'   in layout.keys(): insight.title = layout['title']
	if 'barmode' in layout.keys() and isinstance(layout['barmode'], str): 
		insight.chart_descriptions['barmode'] = layout['barmode'] # stack, group, overlay, ...
	if 'xaxis'   in layout.keys() and 'title' in layout['xaxis']: x_title = layout['xaxis']['title']
	if 'yaxis'   in layout.keys() and 'title' in layout['yaxis']: y_title = layout['yaxis']['title']

	# 先丢弃有多个x或y轴的，或者angular-axis的
	xaxis, yaxis = [], []
	for k in layout.keys():
		if 'angularaxis' in k:
			if trans_logger: trans_logger.log('[{}] has angular-axis'.format(fid))
			return None
		if 'xaxis' in k: xaxis.append(k)
		if 'yaxis' in k: yaxis.append(k)
	if (len(set(xaxis)) > 1) or (len(set(yaxis)) > 1):
		if trans_logger: trans_logger.log('[{}] multi x-axis or y-axis'.format(fid))
		return None

	trace_num = len(chart_data)
	if trace_num == 0: 
		if trans_logger: trans_logger.log('[{}] no traces'.format(fid))
		return None
	if trace_num > 20: # 丢弃超过20个trace的
		if trans_logger: trans_logger.log('[{}] too many traces'.format(fid))
		return None

	# traverse traces
	x_fields, y_fields = [], []
	xsrcs, ysrcs = [], []
	orientations = ['vertical'] * trace_num
	trace_names = []
	opacity = []
	for trace_id, trace in enumerate(chart_data):
		# filter charts using unneed srcs
		for src in abandon_srcs:
			if src in trace.keys():
				if trans_logger: trans_logger.log('[{}] has src {}'.format(fid, src))
				return None
		xsrc, ysrc = trace.get('xsrc'), trace.get('ysrc')
		if (xsrc is None) or (ysrc is None): continue		# 只有一个维度的bar不超过300个
		xsrcs.append(xsrc)
		ysrcs.append(ysrc)
		x_field = get_field_from_src(fid, xsrc, fields_by_name, uid_name_mapping, uids_by_dict, trans_logger)
		y_field = get_field_from_src(fid, ysrc, fields_by_name, uid_name_mapping, uids_by_dict, trans_logger)
		if (x_field is None) or (y_field is None): continue
		# match length
		if len(x_field['data']) != len(y_field['data']):
			length = min(len(x_field['data']), len(y_field['data']))
			x_field['data'] = x_field['data'][:length]
			y_field['data'] = y_field['data'][:length]
		x_fields.append(x_field)
		y_fields.append(y_field)
		# other trace attributes
		ori  = trace.get('orientation')
		opa  = trace.get('opacity')
		name = trace.get('name')
		if ori and ori=='h': orientations[trace_id] = 'horizontal'
		if opa:	opacity.append(opa)
		else:	opacity.append('')
		if name:
			if not isinstance(name, list): trace_names.append(str(name))
			else: trace_names.append(str(name[0]))
		# else: trace_names.append('trace_{}'.format(trace_id))		# 保证name列表和field列表等长
		else: trace_names.append('')
	if len(set(orientations)) > 1:
		if trans_logger: trans_logger.log('[{}] [multi-trace] corss-trace orientations not match'.format(fid))
		return None
	insight.chart_descriptions['orientation'] = orientations[0]		

	if len(x_fields) == 0:
		if trans_logger: trans_logger.log('[{}] reading trace fields error'.format(fid))
		return None
	
	if (len(set(xsrcs)) == 1) and (len(set(ysrcs)) == 1):	# 同一个trace被多次重复
		trace_num = 1

	# single trace
	if trace_num == 1:
		x_field, y_field = x_fields[0], y_fields[0]
		if x_title and isinstance(x_title, str): x_field['name'] = x_title	# 使用layout里的title作为维度名称
		if y_title and isinstance(y_title, str): y_field['name'] = y_title	# 有时title是一个列表。。无语
		# construct insight
		if orientations[0] == 'vertical':		# 竖直bar的breakdown在x轴
			breakdown, measure = x_field, y_field
			insight.breakdown_channels	= ['x']
			insight.measure_channels	= ['y']
		elif orientations[0] == 'horizontal':	# 水平bar的breakdown在y轴
			breakdown, measure = y_field, x_field
			insight.breakdown_channels	= ['y']
			insight.measure_channels	= ['x']
		# 表格：去除NaN、去除breakdown重复、去除小于等于0、重置index
		df = pd.DataFrame({
			breakdown['name']:	breakdown['data'],
			measure['name']:	measure['data'],
		})
		df = df.dropna()
		try: df = df.groupby(breakdown['name'])[measure['name']].max().reset_index()
		except Exception as e:
			if trans_logger: trans_logger.log('[{}] drop-duplicate error: {}'.format(fid, e))
		try:
			df = df[ df[measure['name']] > 0 ]
			df = df.reset_index().drop('index', axis=1)
		except Exception as e:
			if trans_logger: trans_logger.log('[{}] drop measure<0 error: {}'.format(fid, e))
		# insight
		insight.transformed_table	= df
		insight.breakdown_fields	= [breakdown['name']]
		insight.breakdown_types		= [breakdown['type']]
		insight.measure_fields		= [measure['name']]
		insight.measure_types		= [measure['type']]
		if opacity[0] != '': insight.m_chan_descriptions.append({'opacity': opacity[0]})

	# multiple traces
	else:
		if orientations[0] == 'vertical':		# 竖直bar的breakdown在x轴
			breakdowns, measures = x_fields, y_fields
			insight.breakdown_channels	= ['x', 'color']
			insight.measure_channels	= ['y']
		elif orientations[0] == 'horizontal':	# 水平bar的breakdown在y轴
			breakdowns, measures = y_fields, x_fields
			insight.breakdown_channels	= ['y', 'color']
			insight.measure_channels	= ['x']
		color_data, breakdown_data, measure_data = [], [], []
		breakdown_names, breakdown_types, measure_types = [], [], []
		for trace_id in range(len(breakdowns)):
			breakdown_field, measure_field = breakdowns[trace_id], measures[trace_id]
			length = len(x_field['data'])
			name = trace_names[trace_id]
			if name == '': name = measure_field['name']
			color_data.extend([name] * length)
			breakdown_data.append(breakdown_field['data'])
			breakdown_names.append(breakdown_field['name'])
			breakdown_types.append(breakdown_field['type'])
			measure_data.append(measure_field['data'])
			measure_types.append(measure_field['type'])
		if len(set(breakdown_types)) != 1:
			if trans_logger: trans_logger.log('[{}] [multi-trace] breakdown types not consistent'.format(fid))
			return None
		if len(set(measure_types)) != 1:
			if trans_logger: trans_logger.log('[{}] [multi-trace] measure types not consistent'.format(fid))
			return None
		if len(set(breakdown_names)) == 1:
			x_title = breakdown_names[0]
		if x_title=='' or not isinstance(x_title, str): x_title = 'x'
		if y_title=='' or not isinstance(y_title, str): y_title = 'y'
		if orientations[0] == 'vertical':	breakdown_title, measure_title = x_title, y_title
		else:								breakdown_title, measure_title = y_title, x_title
		# 表格：去除NaN、去除小于等于0、重置index
		df = pd.DataFrame({
			'color':			pd.Series(color_data),
			breakdown_title:	pd.concat(breakdown_data, ignore_index=True),
			measure_title:		pd.concat(measure_data, ignore_index=True),
		})
		df = df.dropna()
		try:
			df = df[ df[measure_title] > 0 ]
			df = df.reset_index().drop('index', axis=1)
		except Exception as e:
			if trans_logger: trans_logger.log('[{}] drop measure<0 error: {}'.format(fid, e))
		# abandon: insight 先按C分为多个bar，每个bar再划分为多个色块
		# 还是按和multi-line一样的顺序，先按颜色分trace，再按x轴分bar，和原数据保持一贯的逻辑
		insight.transformed_table	= df
		# insight.breakdown_fields	= [breakdown_title, 'color']
		# insight.breakdown_types		= [breakdown_types[0], 'C']
		insight.breakdown_fields	= ['color', breakdown_title]
		insight.breakdown_types		= ['C', breakdown_types[0]]
		insight.measure_fields		= [measure_title]
		insight.measure_types		= [measure_types[0]]

	insight.transformed_table = insight.transformed_table.dropna()
	if insight.transformed_table.shape[0] == 0:
		if trans_logger: trans_logger.log('[{}] all NaN Error'.format(fid))
		return None
	return insight


def transform_pie_chart_data(chart_obj, fields_by_name, uid_name_mapping, uids_by_dict, trans_logger=None):
	# init
	chart_type	= 'pie'
	fid			= chart_obj.fid
	chart_data	= json.loads(chart_obj.chart_data)
	layout		= json.loads(chart_obj.layout)
	insight = Insight()
	insight.vizml_fid	= fid
	insight.chart_type	= chart_type
	# layout extraction
	if 'title' in layout.keys(): insight.title = layout['title']
	# only use the single-trace pie
	trace_num = len(chart_data)
	if trace_num > 1: 
		if trans_logger: trans_logger.log('[{}] skip multi-traces'.format(fid))
		return None
	# trace info extraction
	trace = chart_data[0]
	# 如果没有title，用trace的name作为title
	if 'name' in trace.keys() and insight.title == '':
		insight.title = trace['name']
	if 'hole' in trace.keys():
		insight.chart_descriptions['hole'] = trace['hole']
		# download_chart(fid, chart_type)
		# draw_chart_plotly_json(chart_obj, chart_type)
	# table data
	valuessrc, labelssrc = trace.get('valuessrc'), trace.get('labelssrc')
	if (valuessrc is None) or (labelssrc is None):
		if trans_logger: trans_logger.log('[{}] lack src'.format(fid))
		return None
	value_field = get_field_from_src(fid, valuessrc, fields_by_name, uid_name_mapping, uids_by_dict, trans_logger)
	label_field = get_field_from_src(fid, labelssrc, fields_by_name, uid_name_mapping, uids_by_dict, trans_logger)
	if (value_field is None) or (label_field is None):
		return None
	# field type constrain
	if value_field['type'] != 'Q':
		if trans_logger: trans_logger.log('[{}] value field type is {}'.format(fid, value_field['type']))
		return None
	# match length
	if len(value_field['data']) != len(label_field['data']):
		length = min(len(value_field['data']), len(label_field['data']))
		value_field['data'] = value_field['data'][:length]
		label_field['data'] = label_field['data'][:length]
	# construct insight
	df = pd.DataFrame({
		label_field['name']: label_field['data'],
		value_field['name']: value_field['data']
	})
	df = df.dropna()							# 去除含NaN的行
	df = df[ df[value_field['name']] > 0 ]		# 去除value小于或等于0的行
	df = df.reset_index().drop('index', axis=1)	# 重置index
	insight.transformed_table = df
	if insight.transformed_table.shape[0] == 0:
		if trans_logger: trans_logger.log('[{}] all NaN or <=0'.format(fid))
		return None
	if insight.transformed_table.shape[0] == 1:
		if trans_logger: trans_logger.log('[{}] has only 1 value'.format(fid))
		return None
	insight.breakdown_channels	= ['label']
	insight.breakdown_fields	= [label_field['name']]
	insight.breakdown_types		= [label_field['type']]
	insight.measure_channels	= ['value']
	insight.measure_fields		= [value_field['name']]
	insight.measure_types		= [value_field['type']]

	return insight


def transform_line_chart_data(chart_obj, fields_by_name, uid_name_mapping, uids_by_dict, trans_logger=None):
	# init
	chart_type	= 'line'
	fid			= chart_obj.fid
	chart_data	= json.loads(chart_obj.chart_data)
	layout		= json.loads(chart_obj.layout)
	insight = Insight()
	insight.vizml_fid	= fid
	insight.chart_type	= chart_type
	abandon_srcs = ['tsrc', 'rsrc', 'zsrc']	# 含有这些src的图表都不要（trace数量分别为1152, 1167, 4）

	# layout extraction
	x_title, y_title = '', ''
	if 'title'   in layout.keys(): insight.title = layout['title']
	if 'xaxis'   in layout.keys() and 'title' in layout['xaxis']: x_title = layout['xaxis']['title']
	if 'yaxis'   in layout.keys() and 'title' in layout['yaxis']: y_title = layout['yaxis']['title']
	
	# 同时有yaxis和yaxis2的是左右双轴，先把些的fid存到一个文件
	yaxis = []
	for k in layout.keys():
		if 'yaxis' in k: yaxis.append(k)
	if len(set(yaxis)) > 1:
		f = open(insight_save_folder+'lines_multi_yaxis.txt', 'a+')
		f.write(fid + '\n')
		f.close()
		if trans_logger: trans_logger.log('[{}] multi y-axis'.format(fid))
		return None

	trace_num = len(chart_data)
	if trace_num == 0: 
		if trans_logger: trans_logger.log('[{}] no traces'.format(fid))
		return None
	if trace_num > 10: # 丢弃超过10个trace的
		if trans_logger: trans_logger.log('[{}] too many traces'.format(fid))
		return None

	# traverse traces
	x_fields, y_fields = [], []
	xsrcs, ysrcs = [], []
	trace_names = []
	for trace_id, trace in enumerate(chart_data):
		# filter charts using unneed srcs
		for src in abandon_srcs:
			if src in trace.keys():
				if trans_logger: trans_logger.log('[{}] has src {}'.format(fid, src))
				return None
		xsrc, ysrc = trace.get('xsrc'), trace.get('ysrc')
		if (xsrc is None) or (ysrc is None): continue		# line 必须x-y都有
		xsrcs.append(xsrc)
		ysrcs.append(ysrc)
		x_field = get_field_from_src(fid, xsrc, fields_by_name, uid_name_mapping, uids_by_dict, trans_logger)
		y_field = get_field_from_src(fid, ysrc, fields_by_name, uid_name_mapping, uids_by_dict, trans_logger)
		if (x_field is None) or (y_field is None): continue
		# match length
		if len(x_field['data']) != len(y_field['data']):
			length = min(len(x_field['data']), len(y_field['data']))
			x_field['data'] = x_field['data'][:length]
			y_field['data'] = y_field['data'][:length]
		# drop duplicated values in x_field (keep the max in y_field)
		df = pd.DataFrame({'x': x_field['data'], 'y': y_field['data']})
		try: df = df.groupby('x')['y'].max().reset_index()
		except Exception as e:
			if trans_logger: trans_logger.log('[{}] drop-duplicate error: {}'.format(fid, e))
		try: df = df.sort_values('x')	# sort by x_field if able
		except Exception as e: pass
		if df.shape[0] <= 1: continue	# drop those has no points or only 1 point
		x_field['data'] = df['x']
		y_field['data'] = df['y']
		x_fields.append(x_field)
		y_fields.append(y_field)
		# trace name
		name = trace.get('name')
		if name:
			if not isinstance(name, list): trace_names.append(str(name))
			else: trace_names.append(str(name[0]))
		# else: trace_names.append('trace_{}'.format(trace_id))		# 保证name列表和field列表等长
		else: trace_names.append('')

	if len(x_fields) == 0:
		if trans_logger: trans_logger.log('[{}] reading trace fields error'.format(fid))
		return None
	
	if (len(set(xsrcs)) == 1) and (len(set(ysrcs)) == 1):	# 同一个trace被多次重复
		trace_num = 1

	# single trace
	if trace_num == 1:
		x_field, y_field = x_fields[0], y_fields[0]
		if x_title and isinstance(x_title, str): x_field['name'] = x_title	# 使用layout里的title作为维度名称
		if y_title and isinstance(y_title, str): y_field['name'] = y_title	# 有时title是一个列表。。无语
		# construct insight
		insight.transformed_table	= pd.DataFrame({
			x_field['name']: x_field['data'],
			y_field['name']: y_field['data'],
		})
		insight.breakdown_channels	= ['x']
		insight.breakdown_fields	= [x_field['name']]
		insight.breakdown_types		= [x_field['type']]
		insight.measure_channels	= ['y']
		insight.measure_fields		= [y_field['name']]
		insight.measure_types		= [y_field['type']]
		
	# single src
	# if len(set(xsrcs)) == 1:
	# 	pass					# able to forming multiple y tables

	# multiple traces
	else:
		color_data, x_data, y_data = [], [], []
		x_names, x_types, y_types = [], [], []
		for trace_id in range(len(x_fields)):
			x_field, y_field = x_fields[trace_id], y_fields[trace_id]
			length = len(x_field['data'])
			name = trace_names[trace_id]
			if name == '': name = y_field['name']
			color_data.extend([name] * length)
			x_data.append(x_field['data'])
			y_data.append(y_field['data'])
			x_names.append(x_field['name'])
			x_types.append(x_field['type'])
			y_types.append(y_field['type'])
		if len(set(x_types)) != 1:
			if trans_logger: trans_logger.log('[{}] [multi-trace] x types not consistent'.format(fid))
			return None
		if len(set(y_types)) != 1:
			if trans_logger: trans_logger.log('[{}] [multi-trace] y types not consistent'.format(fid))
			return None
		if len(set(x_names)) == 1:
			x_title = x_names[0]
		if x_title=='' or not isinstance(x_title, str): x_title = 'x'
		if y_title=='' or not isinstance(y_title, str): y_title = 'y'
		# construct insight
		insight.transformed_table	= pd.DataFrame({
			'color':	pd.Series(color_data),
			x_title:	pd.concat(x_data, ignore_index=True),
			y_title:	pd.concat(y_data, ignore_index=True),
		})
		# 先按C分为多条折线，每条线再根据当地资源的信息绘制
		insight.breakdown_fields	= ['color', x_title]
		insight.breakdown_types		= ['C', x_types[0]]
		insight.measure_fields		= [y_title]
		insight.measure_types		= [y_types[0]]

	insight.transformed_table = insight.transformed_table.dropna()
	if insight.transformed_table.shape[0] == 0:
		if trans_logger: trans_logger.log('[{}] all NaN Error'.format(fid))
		return None
	return insight


def transform_scatter_chart_data(chart_obj, fields_by_name, uid_name_mapping, uids_by_dict, trans_logger=None):
	# init
	chart_type	= 'scatter'
	fid			= chart_obj.fid
	chart_data	= json.loads(chart_obj.chart_data)
	layout		= json.loads(chart_obj.layout)
	insight = Insight()
	insight.vizml_fid	= fid
	insight.chart_type	= chart_type
	abandon_srcs = ['tsrc', 'rsrc', 'zsrc', 'valuesrc']	# 含有这些src的图表都不要（trace数量分别为126, 113, 9, 1）

	# layout extraction
	x_title, y_title = '', ''
	if 'title' in layout.keys(): insight.title = layout['title']
	if 'xaxis' in layout.keys() and 'title' in layout['xaxis']: x_title = layout['xaxis']['title']
	if 'yaxis' in layout.keys() and 'title' in layout['yaxis']: y_title = layout['yaxis']['title']
	
	# 先丢弃有左右多个x或y轴的
	xaxis, yaxis = [], []
	for k in layout.keys():
		if 'xaxis' in k: xaxis.append(k)
		if 'yaxis' in k: yaxis.append(k)
	if (len(set(xaxis)) > 1) or (len(set(yaxis)) > 1):
		if trans_logger: trans_logger.log('[{}] multi x-axis or y-axis'.format(fid))
		return None

	trace_num = len(chart_data)
	if trace_num == 0: 
		if trans_logger: trans_logger.log('[{}] no traces'.format(fid))
		return None
	if trace_num > 20: # 丢弃超过20个trace的，14个trace的有9400个
		if trans_logger: trans_logger.log('[{}] too many traces'.format(fid))
		return None

	# traverse traces
	x_fields, y_fields = [], []
	xsrcs, ysrcs = [], []
	trace_names = []
	for trace_id, trace in enumerate(chart_data):
		# filter charts using unneed srcs
		for src in abandon_srcs:
			if src in trace.keys():
				if trans_logger: trans_logger.log('[{}] has src {}'.format(fid, src))
				return None
		xsrc, ysrc = trace.get('xsrc'), trace.get('ysrc')
		if (xsrc is None) or (ysrc is None): continue		# scatter 必须x-y都有
		xsrcs.append(xsrc)
		ysrcs.append(ysrc)
		x_field = get_field_from_src(fid, xsrc, fields_by_name, uid_name_mapping, uids_by_dict, trans_logger)
		y_field = get_field_from_src(fid, ysrc, fields_by_name, uid_name_mapping, uids_by_dict, trans_logger)
		if (x_field is None) or (y_field is None): continue
		# match length
		if len(x_field['data']) != len(y_field['data']):
			length = min(len(x_field['data']), len(y_field['data']))
			x_field['data'] = x_field['data'][:length]
			y_field['data'] = y_field['data'][:length]
		x_fields.append(x_field)
		y_fields.append(y_field)
		# trace name
		name = trace.get('name')
		if name:
			if not isinstance(name, list): trace_names.append(str(name))
			else: trace_names.append(str(name[0]))
		# else: trace_names.append('trace_{}'.format(trace_id))		# 保证name列表和field列表等长
		else: trace_names.append('')

	if len(x_fields) == 0:
		if trans_logger: trans_logger.log('[{}] reading trace fields error'.format(fid))
		return None
	
	# if (len(set(xsrcs)) == 1) and (len(set(ysrcs)) != 1):	# single x-src，大概是line吧
	# 	if trans_logger: trans_logger.log('[{}] has single x-src'.format(fid))	# user 'automata' 有超过20000张图是line
	# 	return None																# 其他的有几千是散点

	if (len(set(xsrcs)) == 1) and (len(set(ysrcs)) == 1):	# 同一个trace被多次重复
		trace_num = 1

	# single trace
	if trace_num == 1:
		x_field, y_field = x_fields[0], y_fields[0]
		if x_title and isinstance(x_title, str): x_field['name'] = x_title	# 使用layout里的title作为维度名称
		if y_title and isinstance(y_title, str): y_field['name'] = y_title	# 有时title是一个列表。。无语
		# construct insight
		insight.transformed_table	= pd.DataFrame({
			x_field['name']: x_field['data'],
			y_field['name']: y_field['data'],
		})
		insight.breakdown_channels	= ['*']		# TODO
		insight.breakdown_fields	= ['*']
		insight.breakdown_types		= ['*']
		insight.measure_channels	= ['x', 'y']
		insight.measure_fields		= [x_field['name'], y_field['name']]
		insight.measure_types		= [x_field['type'], y_field['type']]

	# multiple traces
	else:
		color_data, x_data, y_data = [], [], []
		x_names, x_types, y_types = [], [], []
		for trace_id in range(len(x_fields)):
			x_field, y_field = x_fields[trace_id], y_fields[trace_id]
			length = len(x_field['data'])
			name = trace_names[trace_id]
			if name == '': name = y_field['name']
			color_data.extend([name] * length)
			x_data.append(x_field['data'])
			y_data.append(y_field['data'])
			x_names.append(x_field['name'])
			x_types.append(x_field['type'])
			y_types.append(y_field['type'])
		if len(set(x_types)) != 1:
			if trans_logger: trans_logger.log('[{}] [multi-trace] x types not consistent'.format(fid))
			return None
		if len(set(y_types)) != 1:
			if trans_logger: trans_logger.log('[{}] [multi-trace] y types not consistent'.format(fid))
			return None
		if len(set(x_names)) == 1:
			x_title = x_names[0]
		if x_title=='' or not isinstance(x_title, str): x_title = 'x'
		if y_title=='' or not isinstance(y_title, str): y_title = 'y'
		# construct insight
		insight.transformed_table	= pd.DataFrame({
			'color':	pd.Series(color_data),
			x_title:	pd.concat(x_data, ignore_index=True),
			y_title:	pd.concat(y_data, ignore_index=True),
		})
		insight.breakdown_fields	= ['color']
		insight.breakdown_types		= ['C']
		insight.measure_fields		= [x_title, y_title]
		insight.measure_types		= [x_types[0], y_types[0]]

	insight.transformed_table = insight.transformed_table.dropna()
	if insight.transformed_table.shape[0] == 0:
		if trans_logger: trans_logger.log('[{}] all NaN Error'.format(fid))
		return None
	return insight


def transform_heatmap_chart_data(chart_obj, fields_by_name, uid_name_mapping, uids_by_dict, trans_logger=None):
	# init
	chart_type	= 'heatmap'
	fid			= chart_obj.fid
	chart_data	= json.loads(chart_obj.chart_data)
	layout		= json.loads(chart_obj.layout)
	insight = Insight()
	insight.vizml_fid	= fid
	insight.chart_type	= chart_type
	abandon_srcs = ['ksrc', 'bsrc', 'customdatasrc']	# 含有这些src的图表都不要（trace数量分别为1, 1, 1）

	# layout extraction
	if 'title' in layout.keys(): insight.title = layout['title']
	if 'xaxis'   in layout.keys() and 'title' in layout['xaxis']: x_title = layout['xaxis']['title']
	if 'yaxis'   in layout.keys() and 'title' in layout['yaxis']: y_title = layout['yaxis']['title']

	# 先丢弃有多个x或y轴的
	xaxis, yaxis = [], []
	for k in layout.keys():
		if 'xaxis' in k: xaxis.append(k)
		if 'yaxis' in k: yaxis.append(k)
	if (len(set(xaxis)) > 1) or (len(set(yaxis)) > 1):
		if trans_logger: trans_logger.log('[{}] multi x-axis or y-axis'.format(fid))
		return None

	# only use the single-trace heatmap
	trace_num = len(chart_data)
	if trace_num > 1: 
		if trans_logger: trans_logger.log('[{}] skip multi-traces'.format(fid))
		return None
	
	# trace info extraction
	trace = chart_data[0]
	# 如果没有title，用trace的name作为title
	if ('name' in trace.keys()) and (insight.title == ''): insight.title = trace['name']

	# zsrc 是一个矩阵，检查每长度是否正确（每个子列表长度和y一样，子列表数量等于x长度）
	# table data
	for src in abandon_srcs:
		if src in trace.keys():
			if trans_logger: trans_logger.log('[{}] has src {}'.format(fid, src))
			return None
	xsrc, ysrc, zsrc = trace.get('xsrc'), trace.get('ysrc'), trace.get('zsrc')
	if zsrc is None:
		if trans_logger: trans_logger.log('[{}] lack z-src'.format(fid))
		return None
	x_field, y_field = None, None
	if xsrc: x_field = get_field_from_src(fid, xsrc, fields_by_name, uid_name_mapping, uids_by_dict, trans_logger)
	if ysrc: y_field = get_field_from_src(fid, ysrc, fields_by_name, uid_name_mapping, uids_by_dict, trans_logger)
	z_field = get_field_from_src(fid, zsrc, fields_by_name, uid_name_mapping, uids_by_dict, trans_logger)
	if z_field is None:
		if trans_logger: trans_logger.log('[{}] read z-src error'.format(fid))
		return None
	if not 'matrix' in z_field.keys():
		if trans_logger: trans_logger.log('[{}] z-src is not a matrix'.format(fid))
		return None
	
	# 没有x或y轴的，用range(len)填充
	y_length, x_length = z_field['matrix'].shape[0], z_field['matrix'].shape[1]
	if x_field and len(x_field['data']) != x_length:
		if trans_logger: trans_logger.log('[{}] x-field length not match'.format(fid))
		return None
	if y_field and len(y_field['data']) != y_length:
		if trans_logger: trans_logger.log('[{}] y-field length not match'.format(fid))
		return None
	if x_field is None:
		x_field = {}
		x_field['data'] = pd.Series(range(1, 1 + x_length)).map(str)
		x_field['name'] = 'x'
		x_field['type'] = 'C'
	if y_field is None:
		y_field = {}
		y_field['data'] = pd.Series(range(1, 1 + y_length)).map(str)
		y_field['name'] = 'y'
		y_field['type'] = 'C'
	
	# # match length
	# if len(value_field['data']) != len(label_field['data']):
	# 	length = min(len(value_field['data']), len(label_field['data']))
	# 	value_field['data'] = value_field['data'][:length]
	# 	label_field['data'] = label_field['data'][:length]
	
	# table
	x_data = []
	for x_i in range(x_length):
		x_data.extend( [ x_field['data'][x_i] ] * y_length )
	y_data = y_field['data'].tolist() * x_length
	# construct insight
	insight.transformed_table = pd.DataFrame({
		x_field['name']: x_data,
		y_field['name']: y_data,
		z_field['name']: z_field['data']
	})
	insight.transformed_table = insight.transformed_table.dropna()
	if insight.transformed_table.shape[0] == 0:
		if trans_logger: trans_logger.log('[{}] all NaN Error'.format(fid))
		return None
	insight.breakdown_channels	= ['x', 'y']
	insight.breakdown_fields	= [x_field['name'], y_field['name']]
	insight.breakdown_types		= [x_field['type'], y_field['type']]
	insight.measure_channels	= ['z']
	insight.measure_fields		= [z_field['name']]
	insight.measure_types		= [z_field['type']]

	return insight


def transform_box_chart_data(chart_obj, fields_by_name, uid_name_mapping, uids_by_dict, trans_logger=None):
	pass


def transform_histogram_chart_data(chart_obj, fields_by_name, uid_name_mapping, uids_by_dict, trans_logger=None):
	# init
	chart_type	= 'histogram'
	fid			= chart_obj.fid
	chart_data	= json.loads(chart_obj.chart_data)
	layout		= json.loads(chart_obj.layout)
	insight = Insight()
	insight.vizml_fid	= fid
	insight.chart_type	= chart_type

	# layout extraction
	x_title, y_title = '', ''
	if 'title' in layout.keys(): insight.title = layout['title']
	if 'xaxis' in layout.keys() and 'title' in layout['xaxis']: x_title = layout['xaxis']['title']
	if 'yaxis' in layout.keys() and 'title' in layout['yaxis']: y_title = layout['yaxis']['title']
	
	# 先丢弃有左右多个x或y轴的
	xaxis, yaxis = [], []
	for k in layout.keys():
		if 'xaxis' in k: xaxis.append(k)
		if 'yaxis' in k: yaxis.append(k)
	if (len(set(xaxis)) > 1) or (len(set(yaxis)) > 1):
		if trans_logger: trans_logger.log('[{}] multi x-axis or y-axis'.format(fid))
		return None

	trace_num = len(chart_data)
	if trace_num == 0: 
		if trans_logger: trans_logger.log('[{}] no traces'.format(fid))
		return None
	if trace_num > 5: # 丢弃超过5个trace的，5个trace的有106个
		if trans_logger: trans_logger.log('[{}] too many traces'.format(fid))
		return None

	# traverse traces
	x_fields, xsrcs, trace_names = [], [], []
	for trace_id, trace in enumerate(chart_data):
		xsrc, ysrc = trace.get('xsrc'), trace.get('ysrc')
		if xsrc is None: continue			# 只要x
		if ysrc is not None: continue
		xsrcs.append(xsrc)
		x_field = get_field_from_src(fid, xsrc, fields_by_name, uid_name_mapping, uids_by_dict, trans_logger)
		if x_field is None: continue
		x_fields.append(x_field)
		# trace name
		name = trace.get('name')
		if name:
			if not isinstance(name, list): trace_names.append(str(name))
			else: trace_names.append(str(name[0]))
		# else: trace_names.append('trace_{}'.format(trace_id))		# 保证name列表和field列表等长
		else: trace_names.append('')

	if len(x_fields) == 0:
		if trans_logger: trans_logger.log('[{}] reading trace fields error'.format(fid))
		return None

	# single trace
	if trace_num == 1:
		x_field = x_fields[0]
		if x_title and isinstance(x_title, str): x_field['name'] = x_title	# 使用layout里的title作为维度名称
		# construct insight
		insight.transformed_table	= pd.DataFrame({
			x_field['name']: x_field['data'],
		})
		insight.breakdown_channels	= ['*']
		insight.breakdown_fields	= ['*']
		insight.breakdown_types		= ['*']
		insight.measure_channels	= ['x']
		insight.measure_fields		= [x_field['name']]
		insight.measure_types		= [x_field['type']]
	
	# multiple traces
	else:
		color_data, x_data = [], []
		x_names, x_types = [], []
		for trace_id in range(len(x_fields)):
			x_field = x_fields[trace_id]
			length = len(x_field['data'])
			name = trace_names[trace_id]
			if name == '': name = 'trace_{}'.format(trace_id)
			color_data.extend([name] * length)
			x_data.append(x_field['data'])
			x_names.append(x_field['name'])
			x_types.append(x_field['type'])
		if len(set(x_types)) != 1:
			if trans_logger: trans_logger.log('[{}] [multi-trace] x types not consistent'.format(fid))
			return None
		if len(set(x_names)) == 1:
			x_title = x_names[0]
		if x_title == '': x_title = 'x'
		# construct insight
		insight.transformed_table	= pd.DataFrame({
			'color':	pd.Series(color_data),
			x_title:	pd.concat(x_data, ignore_index=True),
		})
		insight.breakdown_fields	= ['color']
		insight.breakdown_types		= ['C']
		insight.measure_fields		= [x_title]
		insight.measure_types		= [x_types[0]]

	insight.transformed_table = insight.transformed_table.dropna()
	if insight.transformed_table.shape[0] == 0:
		if trans_logger: trans_logger.log('[{}] all NaN Error'.format(fid))
		return None
	return insight




def main_process(chart_type, trans_logger=None):
	if trans_logger:
		trans_logger.log('')
		trans_logger.log('-------------------------------------------------')
		trans_logger.log(chart_type)
		
	# get raw chunk files
	def chunks_order_info(name):
		chart_type, _, num =  name.split('_')
		num = int(num.split('.')[0])
		return (chart_type, num)
	chunk_files = list_dir(raw_chunks_folder)
	if chart_type == 'scatter':
		chunk_files = [f for f in chunk_files if 'real_scatter' in f]
	else:
		chunk_files = [f for f in chunk_files if chart_type in f]
	chunk_files = sorted(chunk_files, key=chunks_order_info)
	# chunk_files = chunk_files[29:]

	# if chart_type == 'scatter':
		# sca_to_lines = np.loadtxt(insight_save_folder + 'scatter_to_line_Temporal_bug.txt', dtype='str')
		# real_scatters = np.loadtxt(insight_save_folder + 'real_scatter.txt', dtype='str')

# start traversing
	single_insight_df = None	# single-breakdown bar, line, scatter and pie, box (@TODO: histogram and heatmap)
	multi_insight_df  = None	# multi-breakdown bar, line, scatter (multi breakdown fields)
	single_chunk_num = 0
	multi_chunk_num = 0
	failure_num = 0
	insight_statistics = {}
	bar_mode_statisticss = {'group':0, 'stack':0, 'overlay':0}
	bar_orien_statistics = {'vertical':0, 'horizontal':0}
	pie_hole_statistics = []
	for chunk_file in tqdm(chunk_files):
		chunk_df = pd.read_csv(raw_chunks_folder + chunk_file)
		for chart_idx, chart_obj in chunk_df.iterrows():
			fid = chart_obj.fid
			if fid in black_list: continue
			table_data	= json.loads(chart_obj.table_data)
			fields_by_name, uid_name_mapping, uids_by_dict = extract_table_fields(fid, table_data, trans_logger)
			
			if chart_type == 'bar': 
				insight = transform_bar_chart_data(chart_obj, fields_by_name, uid_name_mapping, uids_by_dict, trans_logger)
			elif chart_type == 'pie':
				insight = transform_pie_chart_data(chart_obj, fields_by_name, uid_name_mapping, uids_by_dict, trans_logger)
			elif chart_type == 'line':
				insight = transform_line_chart_data(chart_obj, fields_by_name, uid_name_mapping, uids_by_dict, trans_logger)
			elif chart_type == 'scatter':
				if fid.startswith('automata'): continue
				# if not fid in real_scatters: continue
				insight = transform_scatter_chart_data(chart_obj, fields_by_name, uid_name_mapping, uids_by_dict, trans_logger)
			elif chart_type == 'heatmap':
				insight = transform_heatmap_chart_data(chart_obj, fields_by_name, uid_name_mapping, uids_by_dict, trans_logger)
			elif chart_type == 'box':
				insight = transform_box_chart_data(chart_obj, fields_by_name, uid_name_mapping, uids_by_dict, trans_logger)
			elif chart_type == 'histogram':
				insight = transform_histogram_chart_data(chart_obj, fields_by_name, uid_name_mapping, uids_by_dict, trans_logger)

			if insight == None:
				failure_num += 1
				continue
		
		# statistics
			insight_type = '_'.join(insight.breakdown_types) + '-' + '_'.join(insight.measure_types)
			# if insight_type == '*-C_Q':
			# 	download_chart(fid, chart_type)
			# 	draw_chart_plotly_json(chart_obj, chart_type)
			if not insight_type in insight_statistics:	insight_statistics[insight_type] = 1
			else:										insight_statistics[insight_type] += 1
			if chart_type == 'bar':
				if 'barmode' in insight.chart_descriptions:
					if not insight.chart_descriptions['barmode'] in bar_mode_statisticss.keys():
						bar_mode_statisticss[insight.chart_descriptions['barmode']] = 1
					else: bar_mode_statisticss[insight.chart_descriptions['barmode']] += 1
				if 'orientation' in insight.chart_descriptions:
					if not insight.chart_descriptions['orientation'] in bar_orien_statistics.keys():
						bar_orien_statistics[insight.chart_descriptions['orientation']] = 1
					bar_orien_statistics[insight.chart_descriptions['orientation']] += 1
			if chart_type == 'pie':
				if 'hole' in insight.chart_descriptions:
					pie_hole_statistics.append(insight.chart_descriptions['hole'])

		# transform insight for saving
			# TODO: add chart_index 方便找到在raw_chunk里的位置
			try: insight_df_row = insight.vizml_output_to_one_row()
			except Exception as e:
				if trans_logger: trans_logger.log('[{}] {}'.format(fid, e))
				failure_num += 1
				continue
			if (chart_type=='scatter' and insight.breakdown_fields != ['*']) or (chart_type!='scatter' and len(insight.breakdown_fields) > 1):
				if multi_insight_df is None: multi_insight_df = insight_df_row
				else: multi_insight_df = pd.concat([multi_insight_df, insight_df_row], ignore_index=True)
			else:
				if single_insight_df is None: single_insight_df = insight_df_row
				else: single_insight_df = pd.concat([single_insight_df, insight_df_row], ignore_index=True)
			# save
			if single_insight_df is not None and single_insight_df.shape[0] == chunk_size:
				single_chunk_num += 1
				save_file = insight_save_folder + '{}_single_{}.csv'.format(chart_type, single_chunk_num)
				single_insight_df.to_csv(save_file, index=False)
				del single_insight_df
				single_insight_df = None
				if trans_logger: trans_logger.log('save one chunk to: {}'.format(save_file))
			if multi_insight_df is not None and multi_insight_df.shape[0] == chunk_size:
				multi_chunk_num += 1
				save_file = insight_save_folder + '{}_multi_{}.csv'.format(chart_type, multi_chunk_num)
				multi_insight_df.to_csv(save_file, index=False)
				del multi_insight_df
				multi_insight_df = None
				if trans_logger: trans_logger.log('save one chunk to: {}'.format(save_file))
	
		# finish one raw chunk
		if trans_logger:
			trans_logger.log('Finish Raw Chunk: {}'.format(chunk_file))
			trans_logger.log('')

# save the rests
	single_rest = 0
	multi_rest = 0
	if single_insight_df is not None:
		single_rest = single_insight_df.shape[0]
		save_file = insight_save_folder + '{}_single_{}.csv'.format(chart_type, single_chunk_num+1)
		single_insight_df.to_csv(save_file, index=False)
		if trans_logger:
			trans_logger.log('save last chunk to: {}'.format(save_file))
			trans_logger.log('')
	if multi_insight_df is not None:
		multi_rest = multi_insight_df.shape[0]
		save_file = insight_save_folder + '{}_multi_{}.csv'.format(chart_type, multi_chunk_num+1)
		multi_insight_df.to_csv(save_file, index=False)
		if trans_logger:
			trans_logger.log('save last chunk to: {}'.format(save_file))
			trans_logger.log('')

	if trans_logger:
		trans_logger.log('')
		trans_logger.log('Finish')
		trans_logger.log('{} insights statistics'.format(chart_type))
		trans_logger.log('Failure num: {}'.format(failure_num))
		if chart_type == 'bar':
			trans_logger.log('barmode statistics: ')
			trans_logger.log_dict(bar_mode_statisticss)
			trans_logger.log('orientation statistics: ')
			trans_logger.log_dict(bar_orien_statistics)
		elif chart_type == 'pie':
			trans_logger.log('donut num: {}'.format(len(pie_hole_statistics)))
			val, counts = np.unique(pie_hole_statistics, return_counts=True)
			trans_logger.log('hole values: ')
			trans_logger.log_dict(dict(zip(val, counts)))
		trans_logger.log('single-breakdown num: {}'.format(chunk_size * single_chunk_num + single_rest))
		trans_logger.log('multi-breakdown num: {}'.format(chunk_size * multi_chunk_num + multi_rest))
		trans_logger.log('insight types statistics: ')
		trans_logger.log_dict(insight_statistics)
		trans_logger.log('')


def extract_lines_from_scatter(ex_logger):
	ex_logger.log('Find lines in scatter')
	# get raw chunk files
	def chunks_order_info(name):
		chart_type, _, num =  name.split('_')
		num = int(num.split('.')[0])
		return (chart_type, num)
	chunk_files = list_dir(raw_chunks_folder)
	chunk_files = [f for f in chunk_files if 'scatter_chunk' in f]
	chunk_files = sorted(chunk_files, key=chunks_order_info)
	# scatter_to_line_df = pd.DataFrame(columns=['fid', 'chart_data', 'layout', 'table_data'])
	real_scatter_df = pd.DataFrame(columns=['fid', 'chart_data', 'layout', 'table_data'])
	real_scatter_df_num = 0
	scatter_to_line_num = 0
	for chunk_file in tqdm(chunk_files):
		chunk_df = pd.read_csv(raw_chunks_folder + chunk_file)
		for chart_idx, chart_obj in chunk_df.iterrows():
			fid = chart_obj.fid
			if fid in black_list: continue
			table_data	= json.loads(chart_obj.table_data)
			chart_data	= json.loads(chart_obj.chart_data)
			layout		= json.loads(chart_obj.layout)
			fields_by_name, uid_name_mapping, uids_by_dict = extract_table_fields(fid, table_data, ex_logger)
			trace_num = len(chart_data)
			x_fields, y_fields = [], []
			is_line = True
			for trace_id, trace in enumerate(chart_data):
				xsrc, ysrc = trace.get('xsrc'), trace.get('ysrc')
				mode = trace.get('mode')
				if mode and mode=='markers': is_line = False
				if (xsrc is None) or (ysrc is None): continue		# line 和 scatter 都必须有x-y
				x_field = get_field_from_src(fid, xsrc, fields_by_name, uid_name_mapping, uids_by_dict, trans_logger)
				y_field = get_field_from_src(fid, ysrc, fields_by_name, uid_name_mapping, uids_by_dict, trans_logger)
				if (x_field is None) or (y_field is None): continue
				# match length
				if len(x_field['data']) != len(y_field['data']):
					length = min(len(x_field['data']), len(y_field['data']))
					x_field['data'] = x_field['data'][:length]
					y_field['data'] = y_field['data'][:length]
				x_fields.append(x_field)
				y_fields.append(y_field)
			if len(x_fields) == 0: continue
			
			# trace里没有mode=markers，且y是Q，且x是unique，x是C或严格单调递增的Q/T
			if is_line:
				for y_field in y_fields:
					if y_field['type'] != 'Q':
						is_line = False
			if is_line:
				for x_field in x_fields:
					if (x_field['data'].dropna().value_counts() != 1).any():
						is_line = False
						break
					arr = np.array(x_field['data'].dropna())
					if len(arr) == 0:
						is_line = False
						break
					if x_field['type'] == 'Q':
						# if x_field['data'].dtype == 'float':	# 其实也有很多x轴是float的line
						# 	is_line = False
						# 	break
						sub = np.subtract(arr[1:], arr[:-1])
						if not np.all(sub >= 0):	# 严格单调
							is_line = False
							break
					elif x_field['type'] == 'T':
						if not np.all(arr[1:] > arr[:-1]):
							is_line = False
					elif x_field['type'] == 'C':
						pass
					else:
						is_line = False
			
			if is_line:
				scatter_to_line_num += 1
				# download_chart(fid, 'scatter_line')
				# draw_chart_plotly_json(chart_obj, 'scatter_line')
				# f = open(insight_save_folder+'scatter_to_line.txt', 'a+')
				# f.write(fid + '\n')
				# f.close()
				# scatter_to_line_df = scatter_to_line_df.append(chart_obj, ignore_index=True)
			else:
				f = open(insight_save_folder+'real_scatter.txt', 'a+')
				f.write(fid + '\n')
				f.close()
				real_scatter_df = real_scatter_df.append(chart_obj, ignore_index=True)

			if real_scatter_df is not None and real_scatter_df.shape[0] == chunk_size:
				real_scatter_df_num += 1
				save_file = raw_chunks_folder + 'real_scatter_{}.csv'.format(real_scatter_df_num)
				real_scatter_df.to_csv(save_file, index=False)
				del real_scatter_df
				real_scatter_df = pd.DataFrame(columns=['fid', 'chart_data', 'layout', 'table_data'])
				ex_logger.log('save one chunk to: {}'.format(save_file))

	rest_num = 0
	if real_scatter_df is not None and real_scatter_df.shape[0] > 0:
		rest_num = real_scatter_df.shape[0]
		save_file = raw_chunks_folder + 'real_scatter_{}.csv'.format(real_scatter_df_num+1)
		real_scatter_df.to_csv(save_file, index=False)
		ex_logger.log('save last chunk to: {}'.format(save_file))

	ex_logger.log('Total real scatter num: {}'.format(real_scatter_df_num * chunk_size + rest_num))
	ex_logger.log('Scatter to line num: {}'.format(scatter_to_line_num))



if __name__ == '__main__':
	if not os.path.exists(insight_save_folder):
		os.mkdir(insight_save_folder)
	# log setting
	chart_type = 'bar'
	log_suffix = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
	log_file = 'logs/' + 'vizml_insight_extract_{}_{}.txt'.format(chart_type, log_suffix)
	trans_logger = logger(log_file, {'Mission': 'transform vizml charts into insights'})
	# log_file = 'logs/' + 'vizml_scatter_to_line_{}.txt'.format(log_suffix)
	# trans_logger = logger(log_file, {'Mission': 'find lines in scatters'})
	
	main_process(chart_type, trans_logger)

	# extract_lines_from_scatter(trans_logger)
