from tqdm import tqdm
import pandas as pd
import json
import os

def list_dir(folder):
	contents = os.listdir(folder)
	if '.DS_Store' in contents: 
		contents.remove('.DS_Store')
	return sorted(contents)

dataset = 'data2vis'
spec_data_folder = '../data/data2vis/examples/'
raw_data_folder  = '../data/data2vis/examplesdata/'
sub_folders = list_dir(spec_data_folder)

marks = []
mark_types = {}
mark_statistics = []

channel_keys = []
bin_method = []
scale_values = []
aggreagation_method = []
primitiveType_values = []
selected_values = []
any_values = []
timeunit_values = []
pt_num = 0

### traverse data2vis dataset
for sub_f in sub_folders:
	print(sub_f)
	folder = spec_data_folder + sub_f + '/'
	spec_files = list_dir(folder)
	for spec_f in tqdm(spec_files):
		spec_file = folder + spec_f

	### read vega-lite json
		with open(spec_file, 'r') as f:
			spec = json.load(f)
		data_name = spec['data']['url'].split('/')[-1]
		data_file = raw_data_folder + data_name
		
	### read data json
		# df = pd.read_json(data_file)
		# print(df.head())
		
	### statistics of the channels and field types for each mark
		# mark = spec['mark']
		# if not mark in mark_types:
		# 	marks.append(mark)
		# 	mark_types[mark] = 1
		# 	cols = ('count', 'nominal', 'ordinal', 'quantitative', 'temporal')
		# 	statis = pd.DataFrame(columns=cols)
		# 	mark_statistics.append(statis)
		# else:
		# 	mark_types[mark] += 1
		# 	statis = mark_statistics[marks.index(mark)]

		# for channel in spec['encoding'].keys():
		# 	if not channel in statis.index:
		# 		statis.loc[channel] = [1, 0, 0, 0, 0]
		# 	else:
		# 		statis.loc[channel, 'count'] += 1
		# 	field_type = spec['encoding'][channel]['type']
		# 	statis.loc[channel, field_type] += 1

		# mark_statistics[marks.index(mark)] = statis


	### channel specifications
		primitive_channel_num = 0
		for key in spec['encoding']:
			channel = spec['encoding'][key]
			# if 'timeUnit' in channel.keys() and 'bin' in channel.keys():
			# 	print(channel)
			for channel_key in channel:
				if not channel_key in channel_keys: 
					channel_keys.append(channel_key)
				if channel_key == 'aggregate':
					if not channel['aggregate'] in aggreagation_method:
						aggreagation_method.append(channel['aggregate'])
				elif channel_key == 'bin':
					if not channel['bin'] in bin_method:
						bin_method.append(channel['bin'])
				elif channel_key == 'scale':
					if not channel['scale'] in scale_values:
						scale_values.append(channel['scale'])
				elif channel_key == 'primitiveType':
					pt_num += 1
					if not channel['primitiveType'] in primitiveType_values:
						primitiveType_values.append(channel['primitiveType'])
				elif channel_key == 'selected':
					if not channel['selected'] in selected_values:
						selected_values.append(channel['selected'])
				elif channel_key == '_any':
					if not channel['_any'] in any_values:
						any_values.append(channel['_any'])
				elif channel_key == 'timeUnit':
					if not channel['timeUnit'] in timeunit_values:
						timeunit_values.append(channel['timeUnit'])




# for i in range(len(marks)):
# 	print(marks[i], ': ', mark_types[marks[i]])
# 	print(mark_statistics[i])
# 	print()


print(channel_keys)
print(bin_method)
print(scale_values)
print(aggreagation_method)
print(primitiveType_values)
print(selected_values)
print(any_values)
print(timeunit_values)
print(pt_num)