# !/usr/bin/python3
# -*- coding: utf-8 -*-

'''
Remove charts from a user that are likely created from the same dataset. Uses simplified version of feature extraction

Two-stage process:
1) Generate list of fids to keep based on statistical criteria
2) Output deduplicated chart data

Input: raw plot data AND list of FIDs to preserve (e.g. because of user evaluation)
Output: de-duplicated raw plot data
'''

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
sys.path.insert(0, '..')

# from experiment_data.ground_truth_fids_99 import ground_truth_fids_99
# from helpers.features.single_field_features import extract_single_field_features
# fids_to_preserve = [x['fid'] for x in ground_truth_fids_99]
from helpers.logger import logger
from feature_extraction.type_detection import detect_field_type, data_type_to_general_type
from feature_extraction.helpers import parse

input_file_name  = '../data/vizml_ded/original_tsv/plot_data_with_all_fields_and_header.tsv'
output_file_name = '../data/vizml_ded/original_tsv/plot_data_with_all_fields_and_header_deduplicated_one_per_user.tsv'
headers = ['fid', 'chart_data', 'layout', 'table_data']

log_suffix = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
log_file = 'logs/' + 'deduplicate_{}.txt'.format(log_suffix)
ded_logger = logger(log_file, {'Mission': 'deduplicate vizml full data'})
ded_logger.log('\n')

global unique_fids
global unique_fids_file
unique_fids_file = open('../data/vizml_ded/original_tsv/unique_fids.pkl', 'wb')
output_file = csv.writer(open(output_file_name, 'w'), delimiter='\t')
output_file.writerow(headers)

skipped = 0
num_duplicates = 0
CHUNK_SIZE = 1000
ROW_LIMIT = 100

# List of { features: fid }
existing_features = {}
preserved_fids = []
unique_fids = []

charts_without_data = 0
chart_loading_errors = 0


def load_raw_data(data_file, chunk_size=1000):
	ded_logger.log('Loading raw data from ' + data_file)
	df = pd.read_csv(
		data_file,
		sep='\t',
		error_bad_lines=False,
		chunksize=chunk_size,
		encoding='utf-8'
	)
	return df


def clean_chunk(chunk):
	# Filtering
	df_final_rows = []
	errors = 0
	empty_fields = 0
	global charts_without_data
	global chart_loading_errors

	for i, x in chunk.iterrows():
		try:
			chart_data	= json.loads(x.chart_data)
			layout		= json.loads(x.layout)
			table_data	= json.loads(x.table_data)

			# Filter empty fields
			if not (bool(chart_data) and bool(table_data)):
				empty_fields += 1
				charts_without_data += 1
				chart_loading_errors += 1
				continue

			df_final_rows.append({
				'fid': x['fid'],
				'chart_data': chart_data,
				'layout': layout,
				'table_data': table_data
			})

		except Exception as e:
			errors += 1
			ded_logger.log(e)
			continue

	return pd.DataFrame(df_final_rows)


def find_uniques(chunk, chunk_num):
    global skipped
    global num_duplicates
    df = clean_chunk(chunk)
    start_time = time()

    for chart_num, chart_obj in df.iterrows():
        fid = chart_obj.fid
        table_data = chart_obj.table_data

        absolute_chart_num = ((chunk_num - 1) * CHUNK_SIZE) + chart_num
        if absolute_chart_num % 100 == 0:
            ded_logger.log('[Chunk %s][%s] %.1f: %s %s' % (chunk_num, absolute_chart_num, time(
            ) - start_time, fid, 'https://plot.ly/~{0}/{1}'.format(*fid.split(':'))))

        fields = table_data[list(table_data.keys())[0]]['cols']
        fields = sorted(fields.items(), key=lambda x: x[1]['order'])
        num_fields = len(fields)

        # if num_fields > 25:
        #     skipped += 1
        #     continue
        dataset_features = [num_fields]

        for i, (field_name, d) in enumerate(fields):
            # try:
            field_id = d['uid']
            field_order = d['order']
            field_values = d['data']

            field_length = len(field_values)
            field_type, field_scores = detect_field_type(field_values)
            field_general_type = data_type_to_general_type[field_type]

            try:
                v = parse(field_values, field_type, field_general_type)
                # v = np.ma.array(v).compressed()[:ROW_LIMIT]
                v = np.ma.array(v).compressed()

                characteristic = None
                if len(v):
                    if field_general_type in ['c']: characteristic = pd.Series(v).value_counts().idxmax()
                    if field_general_type in ['t']: characteristic = np.max(v)
                    if field_general_type in ['q']: characteristic = np.mean(v)
            except Exception as e:
                ded_logger.log('Error parsing {}: {}'.format(field_name, e))
                continue
            # = np.append(dataset_features, [field_length, field_general_type, characteristic])
            dataset_features.extend(
                [field_length, field_general_type, characteristic])

        stringified_dataset_features = ''.join(
            [str(s) for s in dataset_features])
        if stringified_dataset_features in existing_features.keys():
            num_duplicates += 1
            old_fid = existing_features[stringified_dataset_features]
            new_fid = fid

            # If we have to preserve the new FID but an identical dataset
            # exists, then replace it
            # if new_fid in fids_to_preserve:
            #     preserved_fids.append(new_fid)
            #     existing_features[stringified_dataset_features] = new_fid
            #     continue
        else:
            # if fid in fids_to_preserve:
            #     preserved_fids.append(new_fid)
            existing_features[stringified_dataset_features] = fid

    ded_logger.log('Num skipped: {}'.format(skipped))
    ded_logger.log('Num preserved FIDs: {}'.format(len(preserved_fids)))
    ded_logger.log('Unique FIDs: {}'.format(len(existing_features.values())))
    ded_logger.log('Duplicates: {} ({:.3f})'.format(num_duplicates,
         							(num_duplicates / absolute_chart_num)))


def write_uniques(chunk, chunk_num):
    chunk_rows = []
    start_time = time()

    absolute_chart_num = ((chunk_num - 1) * CHUNK_SIZE)
    if absolute_chart_num % 100 == 0:
        ded_logger.log(
            '[Chunk %s][%s] %.1f' %
            (chunk_num,
             absolute_chart_num,
             time() -start_time))

    for i, x in chunk.iterrows():
        if x.fid in unique_fids:
            chunk_rows.append([
                x.fid,
                x.chart_data,
                x.layout,
                x.table_data
            ])
    output_file.writerows(chunk_rows)


# def parallelize_ded(data_file):
	# df = load_raw_data(data_file)
	# batch_size = multiprocessing.cpu_count()	# n_jobs
	# ded_logger.log('Number of jobs: ' + str(batch_size))
	# ded_logger.log('')
	# # start
	# batch_num	= 1
	# chunk_batch	= []
	# for i, chunk in enumerate(raw_df_chunks):
	# 	chunk_num = i + 1
	# 	chunk_batch.append(chunk)
	# 	if chunk_num == (batch_size * batch_num):
	# 		ded_logger.log('Start batch {} [chunk {} ~ {}]'.format(
	# 						batch_num, chunk_num-batch_size+1, chunk_num))
	# 		pool = multiprocessing.Pool(batch_size)
	# 		batch_start_time = time.time()
	# 		batch_results = pool.map_async(find_uniques, chunk_batch).get(9999999)
	# 		write_batch_uniques(batch_results)
	# 		batch_time_cost = time.time() - batch_start_time
	# 		ded_logger.log('Finish batch {}'.format(batch_num))
	# 		ded_logger.log('Time cost: {:.1f}s'.format(batch_time_cost))
	# 		ded_logger.log('')
	# 		batch_num	+= 1
	# 		chunk_batch	= []
	# 		pool.close()
	# # process left overs
	# if len(chunk_batch) != 0:
	# 	ded_logger.log('Start last batch {} [chunk {} ~ {}]'.format(
	# 				batch_num, batch_size*(batch_num-1)+1, chunk_num))
	# 	pool = multiprocessing.Pool(batch_size)
	# 	batch_start_time = time.time()
	# 	remaining_batch_results = pool.map_async(find_uniques, chunk_batch).get(9999999)
	# 	write_batch_uniques(remaining_batch_results)
	# 	batch_time_cost = time.time() - batch_start_time
	# 	ded_logger.log('Finish last batch {}'.format(batch_num))
	# 	ded_logger.log('Time cost: {:.1f}s'.format(batch_time_cost))
	# 	ded_logger.log('')
	# 	pool.close()





if __name__ == '__main__':

	# parallelize_ded(input_file_name)


    # input_file_name = '../data/plot_data.tsv'
    raw_df_chunks = pd.read_table(
        input_file_name,
        error_bad_lines=False,
        chunksize=1000,
        encoding='utf-8'
    )

    for i, chunk in tqdm(enumerate(raw_df_chunks)):
        r = find_uniques(chunk, i + 1)
    unique_fids = set(existing_features.values())
    pickle.dump(unique_fids, unique_fids_file)

    raw_df_chunks = pd.read_table(
        input_file_name,
        error_bad_lines=False,
        chunksize=1000,
        encoding='utf-8'
    )
    for i, chunk in tqdm(enumerate(raw_df_chunks)):
        r = write_uniques(chunk, i + 1)
