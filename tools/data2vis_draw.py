# import pandas as pd
from tqdm import tqdm
import altair as alt
# import altair_saver
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
save_dir		 = '../data/data2vis/visualizations/'
sub_folders = list_dir(spec_data_folder)

if not os.path.exists(save_dir): os.mkdir(save_dir)

for sub_f in sub_folders:
	print(sub_f)
	folder = spec_data_folder + sub_f + '/'
	spec_files = list_dir(folder)
	for spec_f in tqdm(spec_files):
		spec_file = folder + spec_f
		
		# vega-lite json
		with open(spec_file, 'r') as f:
			spec = json.load(f)
		data_name = spec['data']['url'].split('/')[-1]
		data_file = raw_data_folder + data_name
		
		# data json
		with open(data_file, 'r') as f:
			data = json.load(f)
		# spec['data']['url'] = data_file  '../../examplesdata/'
		spec['data'] = {'values': data}

		# delete validating specifications due to altair
		for key in spec['encoding'].keys():
			if 'scale' in spec['encoding'][key]:
				if 'bandSize' in spec['encoding'][key]['scale']:
					spec['encoding'][key]['scale'].pop('bandSize')
			if 'primitiveType' in spec['encoding'][key]:
				spec['encoding'][key].pop('primitiveType')
			if 'selected' in spec['encoding'][key]:
				spec['encoding'][key].pop('selected')
			if '_any' in spec['encoding'][key]:
				spec['encoding'][key].pop('_any')
		if 'cell' in spec['config'].keys():
			spec['config'].pop('cell')
		if '_info' in spec.keys():
			spec.pop('_info')
		
		# visualize
		describe = ''
		for channel in spec['encoding'].keys():
			describe += channel + ': ' + spec['encoding'][channel]['type'] + '\n'	
		spec_string = json.dumps(spec)
		chart = alt.Chart.from_json(spec_string, validate=False)
		chart.title = describe

		# save
		save_folder = save_dir + data_name.split('.')[0] + '/'
		if not os.path.exists(save_folder):
			os.mkdir(save_folder)
		save_id = len(list_dir(save_folder)) + 1
		save_file = save_folder + data_name.split('.')[0] + '_{}_{}.html'.format(save_id, spec['mark'])
		chart.save(save_file)
		# save_file = save_folder + data_name.split('.')[0] + '_{}.png'.format(save_id)
		# altair_saver.save(chart, save_file)