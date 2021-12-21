from tqdm import tqdm
import pandas as pd
import numpy as np
import requests
import plotly
import copy
import json
import sys
import os

sys.path.insert(0, '..')
from feature_extraction.type_detection import detect_field_type, data_type_to_general_type
from feature_extraction.helpers import parse
from helpers.utils import list_dir

cases_folder = '../data/vizml_ded/cases/'



def download_chart(fid, chart_type, info=None):
	requests.adapters.DEFAULT_RETRIES = 5
	ses = requests.session()
	ses.keep_alive = False

	def vizml_download(url, filename):
		try:
			response = ses.get(url)
			if response.status_code == 200:
				with open(filename, 'wb') as f:
					f.write(response.content)
			else:
				print('download fail for {}: {}'.format(filename.split('/')[-1], response.status_code))
			response.close()
		except Exception as e:
			print('download fail for {}: {}'.format(filename.split('/')[-1], e))
	
	user, idx = fid.split(':')
	chart_folder = cases_folder + chart_type + '/'
	if not os.path.exists(chart_folder):
		os.mkdir(chart_folder)
	url_prefix = 'https://chart-studio.plotly.com/~{}/{}'.format(user, idx)
	filename_prefix = chart_folder + '{}_{}_download'.format(user, idx)
	if info: filename_prefix = filename_prefix + '_{}'.format(info)
	for suffix in ['.png', '.json', '.csv']:
		vizml_download(url_prefix+suffix, filename_prefix+suffix)


def draw_chart_plotly_json(chart_obj, chart_type):
	chart_folder = cases_folder + chart_type + '/'
	if not os.path.exists(chart_folder):
		os.mkdir(chart_folder)
	fid			= chart_obj.fid
	user, idx	= fid.split(':')
	chart_data	= json.loads(chart_obj.chart_data)
	layout		= json.loads(chart_obj.layout)
	table_data	= json.loads(chart_obj.table_data)
	# cleaning
	if '' in layout.keys() and layout['']=='': layout.pop('')
	# save jsons
	with open(chart_folder + '{}_{}_chart.json'.format(user, idx),	'w') as fc: json.dump(chart_data,	fc)
	with open(chart_folder + '{}_{}_layout.json'.format(user, idx),	'w') as fl: json.dump(layout,		fl)
	with open(chart_folder + '{}_{}_table.json'.format(user, idx),	'w') as ft: json.dump(table_data,	ft)
	
	# replace the str 'true' and 'false' into bool
	def fix_dict(v):
		if isinstance(v, list):
			for item in v: item = fix_dict(item)
		if isinstance(v, dict):
			for key in v.keys(): v[key] = fix_dict(v[key])
		else:
			if (v == 'true') or (v == 'True'): return True
			if (v =='false') or (v =='False'): return False
		return v
	chart_data	= fix_dict(chart_data)
	layout		= fix_dict(layout)

	# get fields data
	# fields = table_data[list(table_data.keys())[0]]['cols']
	fields_by_id = {}
	table_data_dict_num = len(table_data.keys())
	for dict_id in range(table_data_dict_num):
		fields = table_data[list(table_data.keys())[dict_id]]['cols']	# 不同组的fields有可能名称相同
		for field_name, d in fields.items():							# 名称相同的field数据也相同，可以合并（除了it_dcm的5330和5332有错误）
			if d['uid'] in fields_by_id.keys():		# 验证不同组的field的uid是否有可能重复，结论：没有重复
				import IPython
				IPython.embed()
			fields_by_id[d['uid']] = d
			fields_by_id[d['uid']]['name'] = field_name
			fields_by_id[d['uid']]['uid'] = d['uid']
			field_type, _ = detect_field_type(d['data'])
			try:	# TODO: temporal data should still be string to draw
				v = parse(d['data'], field_type, drop=False)
				v = np.ma.array(v).compressed()
				fields_by_id[d['uid']]['data'] = v.tolist()
			except Exception as e:
				print(e)
	# replace chart data src with table data
	chart_data_fill = copy.deepcopy(chart_data)
	for trace_id, trace in enumerate(chart_data):
		for trace_key in trace.keys():
			if trace_key[-3:] == 'src':
				data_key = trace_key[:-3]
				data_uid = trace[trace_key].split(':')[-1]	# 出现了一个src后面一长串uid的情况
				if len(data_uid) == 6: data_values = fields_by_id[data_uid]['data']
				elif (len(data_uid)==7) and data_uid.startswith('-'): data_values = fields_by_id[data_uid[1:]]['data']
				else:
					uid_list = data_uid.split(',')
					data_values = []
					for uuid in uid_list:
						if (len(uuid)==7) and uuid.startswith('-'): uuid = uuid[1:]
						if not uuid in fields_by_id.keys(): return None
						data_values.append(fields_by_id[uuid]['data'])
				chart_data_fill[trace_id].pop(trace_key)
				if data_key in chart_data_fill[trace_id].keys():
					if len(chart_data_fill[trace_id][data_key]) == 0:
						chart_data_fill[trace_id].pop(data_key)		# parksjin01的35和43有空的x和y留在这。。
					else:
						import IPython
						IPython.embed()
				chart_data_fill[trace_id][data_key] = data_values

	if chart_type == 'bar':
		for trace_id, trace in enumerate(chart_data_fill):
			if 'z' in trace.keys(): chart_data_fill[trace_id].pop('z')	# bar 不能有z
			# if 'visible' in trace.keys():
			# 	if trace['visible'] == 'True': chart_data_fill[trace_id]['visible'] = True
			# 	elif trace['visible'] == 'False': chart_data_fill[trace_id]['visible'] = False
			if 'mode' in trace.keys(): chart_data_fill[trace_id].pop('mode')
			if 'marker' in trace.keys() and 'line' in trace['marker'].keys() and \
				'color' in trace['marker']['line'].keys() and trace['marker']['line']['color']=='transparent':
				chart_data_fill[trace_id]['marker'].pop('line')
			if 'marker' in trace.keys() and len(trace['marker'])==0: chart_data_fill[trace_id].pop('marker')
	if chart_type == 'line':
		# axis-type: '-', 'linear', 'log', 'date', 'category', 'multicategory'
		if ('xaxis' in layout.keys()) and ('type' in layout['xaxis'].keys()):
			layout['xaxis']['type'] = layout['xaxis']['type'].lower()
		if ('yaxis' in layout.keys()) and ('type' in layout['yaxis'].keys()):
			layout['yaxis']['type'] = layout['yaxis']['type'].lower()
		# 好像这个plotly不支持type==line，只能是scatter然后mode=lines
		for trace_id, trace in enumerate(chart_data_fill):
			if trace['type'] == 'line':
				chart_data_fill[trace_id]['type'] = 'scatter'
				if 'mode' in chart_data_fill[trace_id].keys():
					import IPython
					IPython.embed()
				chart_data_fill[trace_id]['mode'] = 'lines'

	chart_obj_json = {'data': chart_data_fill, 'layout': layout}
	if fid == 'nicolaslambers:94':	# 挽救成功
		chart_data_fill[0]['type'] == 'line'
		layout = {}
		chart_obj_json = {'data': chart_data_fill, 'layout': layout}
	try:
		fig = plotly.io.from_json(json.dumps(chart_obj_json))
		fig.write_image(chart_folder + '{}_{}_plotly.png'.format(user, idx))
	except Exception as e:
		with open(chart_folder + '{}_{}_plotly_error.txt'.format(user, idx), 'w') as f:
			f.write(str(e))


def find_one_case(target_fid, chart_type):
	def list_dir(folder):
		contents = os.listdir(folder)
		if '.DS_Store' in contents: 
			contents.remove('.DS_Store')
		return sorted(contents)

	raw_chunks_folder = '../data/vizml_ded/raw_data/'
	chunk_files = list_dir(raw_chunks_folder)
	chunk_files = sorted([f for f in chunk_files if chart_type in f])
	# chunk_files = chunk_files[27:]
	found = False
	for chunk_file in tqdm(chunk_files):
		chunk_df = pd.read_csv(raw_chunks_folder + chunk_file)
		for chart_idx, chart_obj in chunk_df.iterrows():
			fid = chart_obj.fid
			if fid == target_fid:
				print('found at ' + chunk_file)
				download_chart(fid, chart_type)
				draw_chart_plotly_json(chart_obj, chart_type)
				found = True
				break
		if found:
			break


def look_at_insight():
	def chunks_order_info(name):
		chart_type, _, num =  name.split('_')
		num = int(num.split('.')[0])
		return num
	insight_save_folder	= '../data/{}/insights/'.format(dataset)
	chart_type = 'line_single'
	insight_files = list_dir(insight_save_folder + chart_type)
	insight_files = sorted(insight_files, key=chunks_order_info)
	for insight_file in insight_files:
		insights_chunk = pd.read_csv(insight_files)




if __name__ == '__main__':
	target_fid = '94harshsharma:7'
	chart_type = 'bar'
	find_one_case(target_fid, chart_type)