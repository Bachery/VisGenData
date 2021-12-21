from tqdm import tqdm
from time import time
import pandas as pd
import numpy as np
import json
import os

from scipy.stats import entropy, normaltest, mode, kurtosis, skew, pearsonr, moment, chisquare, kstest
from scipy.stats import f_oneway, chi2_contingency, ks_2samp
import editdistance



dataset = 'data2vis'
spec_data_folder = '../data/data2vis/examples/'
raw_data_folder  = '../data/data2vis/examplesdata/'

def list_dir(folder):
	contents = os.listdir(folder)
	if '.DS_Store' in contents: 
		contents.remove('.DS_Store')
	return sorted(contents)


def extract_view_info(spec_file):
	view_info = {'data':		None,	# a pandas DataFrame
				 'chart':		'', 	# str: area, bar, line, point, circle or tick
				 'channels':	[], 	# list of visual channels: x, y, color, shape, size, detail
				 'fields':		[], 	# list of field names, '*' if use 'count'
				 'types':		[],		# list of field types: quantitative, temporal, nominal, ordinal
				 'pri_types':	[],		# list of primitive field types: string, interger, number, None
				 'transforms':	[],		# list of transforms on the fields, each element is a dict
				 'describe':	[]}		# list of additional description channels, each element is a dict

	### read vega-lite json
	with open(spec_file, 'r') as f:
		spec = json.load(f)
		
	### read data json
	data_name = spec['data']['url'].split('/')[-1]
	data_file = raw_data_folder + data_name
	df = pd.read_json(data_file)
	view_info['data'] = df
	# print(df.head())

	### extract information constructing the view
	view_info['chart'] = spec['mark']
	encodings = spec['encoding']
	for channel in encodings.keys():
		view_info['channels'].append(channel)
		view_info['fields'].append(encodings[channel]['field'])
		view_info['types'].append(encodings[channel]['type'])
		# primitive types
		if 'primitiveType' in encodings[channel].keys():
			view_info['pri_types'].append(encodings[channel]['primitiveType'])
		else:
			view_info['pri_types'].append(None)
		# transform
		if 'bin' in encodings[channel].keys():
			view_info['transforms'].append({'bin': True})
		elif 'aggregate' in encodings[channel].keys():
			agg_type = encodings[channel]['aggregate']			# count or mean
			view_info['transforms'].append({'aggregate': agg_type})
		elif 'timeUnit' in encodings[channel].keys():
			time_unit_type = encodings[channel]['timeUnit']		# year
			view_info['transforms'].append({'timeUnit': time_unit_type})
		else:
			view_info['transforms'].append(None)
		# describe
		if 'scale' in encodings[channel]:
			if 'zero' in encodings[channel]['scale']:
				view_info['describe'].append({'zero': False})	# not start from zero
			else: 
				view_info['describe'].append(None)
		else:
			view_info['describe'].append(None)

	return view_info


def traverse_all():
	### traverse data2vis dataset
	sub_folders = list_dir(spec_data_folder)
	for sub_f in sub_folders[4:]:
		print(sub_f)
		folder = spec_data_folder + sub_f + '/'
		spec_files = list_dir(folder)
		for spec_id, spec_f in tqdm(enumerate(spec_files)):
			spec_file = folder + spec_f
			view_info = extract_view_info(spec_file)

			if sub_f != 'crimea1' or spec_id < 86 or spec_id > 87: continue
			
			# evaluation
			eva = evaluator(view_info)
			# s = eva.cal_breakdown_features()
			# if s['is_unique']: print('{}_{}: unique breakdown'.format(sub_f, spec_id+1))
			# if s['chi_square_q_095'] and s['num_unique_elements'] < 10:
			# 	print('{}_{}: even breakdown with few groups'.format(sub_f, spec_id+1))
			s = eva.cal_measure_features()



###### feature functions #######

def get_uniqueness_features(series):
	# unique_elements = np.unique(v)
	unique_counts = series.value_counts()
	r = {}
	r['num_unique_elements']	= unique_counts.size
	r['unique_percent']			= unique_counts.size /  len(series)
	r['is_unique']				= unique_counts.size == len(series)
	return r


def get_chi_square(series):
	# unique, counts = np.unique(v, return_counts=True)
	unique_counts = series.value_counts()
	chi_square_score, chi_square_q = chisquare(unique_counts.tolist())
	r = {}
	r['chi_square_score']	= chi_square_score
	r['chi_square_q']		= chi_square_q
	r['chi_square_q_095']	= chi_square_q > 0.95
	return r


def get_trend(series):
	pass


def get_proportion_features(series):
	s = series.sort_values()
	max_1 = s[0]
	max_2 = s[1]
	max_3 = s[2]
	r = {}
	r['max_1_take_50'] = max_1 > series.sum()
	r['max_2_take_50'] = max_1+max_2 > series.sum()
	r['max_3_take_50'] = max_1+max_2+max_3 > series.sum()


def get_correlation_qq(series_a, series_b):
	correlation_value, correlation_p = pearsonr(series_a, series_b)
	ks_statistic, ks_p = ks_2samp(series_a, series_b)
	has_overlap, overlap_percent = calculate_overlap(series_a, series_b)

	r['correlation_value'] = correlation_value
	r['correlation_p'] = correlation_p
	r['correlation_significant_005'] = (correlation_p < 0.05)

	r['ks_statistic'] = ks_statistic
	r['ks_p'] = ks_p
	r['ks_significant_005'] = (ks_p < 0.05)

	r['has_overlap'] = has_overlap
	r['overlap_percent'] = overlap_percent



field_q_statistical_features_list = [
    {'name': 'mean', 'type': 'numeric'},
    {'name': 'normalized_mean', 'type': 'numeric'},
    {'name': 'median', 'type': 'numeric'},
    {'name': 'normalized_median', 'type': 'numeric'},

    {'name': 'var', 'type': 'numeric'},
    {'name': 'std', 'type': 'numeric'},
    {'name': 'coeff_var', 'type': 'numeric'},
    {'name': 'min', 'type': 'numeric'},
    {'name': 'max', 'type': 'numeric'},
    {'name': 'range', 'type': 'numeric'},
    {'name': 'normalized_range', 'type': 'numeric'},

    {'name': 'entropy', 'type': 'numeric'},
    {'name': 'gini', 'type': 'numeric'},
    {'name': 'q25', 'type': 'numeric'},
    {'name': 'q75', 'type': 'numeric'},
    {'name': 'med_abs_dev', 'type': 'numeric'},
    {'name': 'avg_abs_dev', 'type': 'numeric'},
    {'name': 'quant_coeff_disp', 'type': 'numeric'},
    {'name': 'skewness', 'type': 'numeric'},            # 偏度，也是三阶标准矩。(-∞,+∞)。For normally distributed data, the skewness should be about 0
    {'name': 'kurtosis', 'type': 'numeric'},            # 峰度，也是四阶标准矩。[1,+∞)。完全正态分布：kurtosis=3
    {'name': 'moment_5', 'type': 'numeric'},            # 矩。
    {'name': 'moment_6', 'type': 'numeric'},
    {'name': 'moment_7', 'type': 'numeric'},
    {'name': 'moment_8', 'type': 'numeric'},
    {'name': 'moment_9', 'type': 'numeric'},
    {'name': 'moment_10', 'type': 'numeric'},

    {'name': 'percent_outliers_15iqr', 'type': 'numeric'},
    {'name': 'percent_outliers_3iqr', 'type': 'numeric'},
    {'name': 'percent_outliers_1_99', 'type': 'numeric'},
    {'name': 'percent_outliers_3std', 'type': 'numeric'},
    {'name': 'has_outliers_15iqr', 'type': 'boolean'},
    {'name': 'has_outliers_3iqr', 'type': 'boolean'},
    {'name': 'has_outliers_1_99', 'type': 'boolean'},
    {'name': 'has_outliers_3std', 'type': 'boolean'},
    {'name': 'normality_statistic', 'type': 'numeric'},
    {'name': 'normality_p', 'type': 'numeric'},
    {'name': 'is_normal_5', 'type': 'boolean'},
    {'name': 'is_normal_1', 'type': 'boolean'},
]
def get_statistical_features_q(series):
	r = dict([(f['name'], None)
						for f in field_q_statistical_features_list])
	v = series.tolist()

	sample_mean = np.mean(v)
	sample_median = np.median(v)
	sample_var = np.var(v)
	sample_min = np.min(v)
	sample_max = np.max(v)
	sample_std = np.std(v)
	q1, q25, q75, q99 = np.percentile(v, [0.01, 0.25, 0.75, 0.99])
	iqr = q75 - q25

	r['mean'] = sample_mean
	r['normalized_mean'] = sample_mean / sample_max
	r['median'] = sample_median
	r['normalized_median'] = sample_median / sample_max

	r['var'] = sample_var
	r['std'] = sample_std
	r['coeff_var'] = (sample_mean / sample_var) if sample_var else None
	r['min'] = sample_min
	r['max'] = sample_max
	r['range'] = r['max'] - r['min']
	r['normalized_range'] = (r['max'] - r['min']) / \
		sample_mean if sample_mean else None

	r['entropy'] = entropy(v)
	# r['gini'] = gini(v)
	r['q25'] = q25
	r['q75'] = q75
	r['med_abs_dev'] = np.median(np.absolute(v - sample_median))
	r['avg_abs_dev'] = np.mean(np.absolute(v - sample_mean))
	r['quant_coeff_disp'] = (q75 - q25) / (q75 + q25)
	r['coeff_var'] = sample_var / sample_mean
	r['skewness'] = skew(v)
	r['kurtosis'] = kurtosis(v)
	r['moment_5'] = moment(v, moment=5)
	r['moment_6'] = moment(v, moment=6)
	r['moment_7'] = moment(v, moment=7)
	r['moment_8'] = moment(v, moment=8)
	r['moment_9'] = moment(v, moment=9)
	r['moment_10'] = moment(v, moment=10)

	# Outliers
	outliers_15iqr = np.logical_or(
		v < (q25 - 1.5 * iqr), v > (q75 + 1.5 * iqr))
	outliers_3iqr = np.logical_or(v < (q25 - 3 * iqr), v > (q75 + 3 * iqr))
	outliers_1_99 = np.logical_or(v < q1, v > q99)
	outliers_3std = np.logical_or(
		v < (
			sample_mean -
			3 *
			sample_std),
		v > (
			sample_mean +
			3 *
			sample_std))
	r['percent_outliers_15iqr'] = np.sum(outliers_15iqr) / len(v)
	r['percent_outliers_3iqr'] = np.sum(outliers_3iqr) / len(v)
	r['percent_outliers_1_99'] = np.sum(outliers_1_99) / len(v)
	r['percent_outliers_3std'] = np.sum(outliers_3std) / len(v)

	r['has_outliers_15iqr'] = np.any(outliers_15iqr)
	r['has_outliers_3iqr'] = np.any(outliers_3iqr)
	r['has_outliers_1_99'] = np.any(outliers_1_99)
	r['has_outliers_3std'] = np.any(outliers_3std)

	# Statistical Distribution
	if len(v) >= 8:
		normality_k2, normality_p = normaltest(v)
		r['normality_statistic'] = normality_k2
		r['normality_p'] = normality_p
		r['is_normal_5'] = (normality_p < 0.05)
		r['is_normal_1'] = (normality_p < 0.01)

	return r


qq_pairwise_features_list = [
    {'name': 'correlation_value',           'type': 'numeric'},     # The Pearson correlation coefficient measures the linear relationship between two datasets
    {'name': 'correlation_p',               'type': 'numeric'},
    {'name': 'correlation_significant_005', 'type': 'boolean'},
    {'name': 'ks_statistic',                'type': 'numeric'},     # a two-sided test for the null hypothesis that 2 independent samples
    {'name': 'ks_p',                        'type': 'numeric'},     # are drawn from the same continuous distribution.
    {'name': 'ks_significant_005',          'type': 'boolean'},    
    {'name': 'percent_range_overlap',       'type': 'numeric'},
    {'name': 'has_range_overlap',           'type': 'numeric'},
]

def calculate_overlap(a_data, b_data):
    a_min, a_max = np.min(a_data), np.max(a_data)
    a_range = a_max - a_min
    b_min, b_max = np.min(b_data), np.max(b_data)
    b_range = b_max - b_min
    has_overlap = False
    overlap_percent = 0
    if (a_max >= b_min) and (b_min >= a_min):
        has_overlap = True
        overlap = (a_max - b_min)
    if (b_max >= a_min) and (a_min >= b_min):
        has_overlap = True
        overlap = (b_max - a_min)
    if has_overlap:
        overlap_percent = max(overlap / a_range, overlap / b_range)
    if ((b_max >= a_max) and (b_min <= a_min)) or (
            (a_max >= b_max) and (a_min <= b_min)):
        has_overlap = True
        overlap_percent = 1
    return has_overlap, overlap_percent

def get_statistical_pairwise_features(a, b, MAX_GROUPS=50):
	r = dict([ (f['name'], None) for f in qq_pairwise_features_list ])

	a_data = a.tolist()
	b_data = b.tolist() 

	# Match lengths
	min_len = min(len(a_data), len(b_data))
	a_data = a_data[:min_len]
	b_data = b_data[:min_len]       

	correlation_value, correlation_p = pearsonr(a_data, b_data)
	ks_statistic, ks_p = ks_2samp(a_data, b_data)
	has_overlap, overlap_percent = calculate_overlap(a_data, b_data)

	r['correlation_value'] = correlation_value
	r['correlation_p'] = correlation_p
	r['correlation_significant_005'] = (correlation_p < 0.05)

	r['ks_statistic'] = ks_statistic
	r['ks_p'] = ks_p
	r['ks_significant_005'] = (ks_p < 0.05)

	r['has_overlap'] = has_overlap
	r['overlap_percent'] = overlap_percent

	return r


class evaluator(object):
	def __init__(self, view_info):
		self.data		=	view_info['data']		# a pandas DataFrame
		self.chart		=	view_info['chart']		# str: area, bar, line, point, circle or tick
		self.channels	=	view_info['channels']	# list of visual channels: x, y, color, shape, size, detail
		self.fields		=	view_info['fields']		# list of field names, '*' if use 'count'
		self.types		=	view_info['types']		# list of field types: quantitative, temporal, nominal, ordinal
		self.pri_types	=	view_info['pri_types']	# list of primitive field types: string, interger, number, None
		self.transforms	=	view_info['transforms']	# list of transforms on the fields, each element is a dict
		self.describe	=	view_info['describe']	# list of additional description channels, each element is a dict
		
		self.data.dropna(axis=0, how='any', inplace=True)

		self.breakdown_features = {}
		self.measure_features = {}

		self.breakdown_fields	= []
		self.breakdown_types	= []
		self.breakdown_channels	= []
		self.measure_fields		= []
		self.measure_types		= []
		self.measure_channels	= []
		for i in range(len(self.fields)):
			if self.types[i] == 'quantitative' and \
				((self.transforms[i] is not None and not 'bin' in self.transforms[i]) \
				or self.transforms[i] is None):
				if self.fields[i] == '*':
					assert self.transforms[i]['aggregate'] == 'count'
					self.measure_fields.append('COUNT')
					self.measure_types.append('COUNT')
				else:
					self.measure_fields.append(self.fields[i])
					if self.transforms[i] is not None and 'aggregate' in self.transforms[i]:
						assert self.transforms[i]['aggregate'] == 'mean'
						self.measure_types.append('MEAN_'.format(self.types[i]))
					else: 
						self.measure_types.append(self.types[i])
				self.measure_channels.append(self.channels[i])
			elif self.types[i] == 'quantitative' and \
				self.transforms[i] is not None and 'bin' in self.transforms[i]:
				self.breakdown_fields.append(self.fields[i])
				self.breakdown_types.append('BIN_{}'.format(self.types[i]))
				self.breakdown_channels.append(self.channels[i])
			else:
				self.breakdown_fields.append(self.fields[i])
				if self.transforms[i] is not None and 'timeUnit' in self.transforms[i]:
					assert self.types[i] == 'temporal'
					assert self.transforms[i]['timeUnit'] == 'year'
					self.breakdown_types.append('YEAR_{}'.format(self.types[i]))
				self.breakdown_types.append(self.types[i])
				self.breakdown_channels.append(self.channels[i])


	def cal_breakdown_features(self):

		def bin_data(series):
			ma = max(series)
			mi = min(series)
			cut = (ma - mi) / 5
			bins = [mi-1, mi+cut, mi+cut*2, mi+cut*3, mi+cut*4, ma]
			binned_data = pd.cut(series, bins)
			return binned_data

		breakdown_num = len(self.breakdown_fields)
		if breakdown_num == 1:
			group_data = self.data[self.breakdown_fields[0]]
			if self.breakdown_types[0] == 'BIN_quantitative':
				group_data = bin_data(group_data)
		else:
			# 合并算unique，还没考虑 TimeUnit(T) C*C的nested
			if self.breakdown_types[0] == 'BIN_quantitative':
				breakdown_1 = bin_data(self.data[self.breakdown_fields[0]])
			else:
				breakdown_1 = self.data[self.breakdown_fields[0]]
			if self.breakdown_types[1] == 'BIN_quantitative':
				breakdown_2 = bin_data(self.data[self.breakdown_fields[1]])
			else:
				breakdown_2 = self.data[self.breakdown_fields[1]]
			group_data = pd.Series(list(zip(breakdown_1, breakdown_2)))
		
		unique_scores = get_uniqueness_features(group_data)
		# if not unique_scores['is_unique']:
		chi_square_scores = get_chi_square(group_data)
		self.breakdown_features.update(unique_scores)
		self.breakdown_features.update(chi_square_scores)

		return self.breakdown_features


	def cal_measure_features(self):
		measure_num = len(self.measure_fields)
		if measure_num == 1:
			return
			if len(self.breakdown_fields) == 1:
				measure_group = self.data.groupby(self.breakdown_fields[0])
				if   self.measure_types[0] == 'COUNT':			return#measure = measure_group.count()
				elif self.measure_types[0].startswith('MEAN'):	measure = measure_group.mean()
				else: return None
				self.breakdown_features = get_statistical_features_q(measure.iloc[:, 0])
				import powerlaw
				fit = powerlaw.Fit(measure.iloc[:, 0])
				alpha = fit.power_law.alpha
				xmin = fit.power_law.xmin
				print(alpha)
				print(xmin)
				ks_score, p = kstest(measure.iloc[:, 0].tolist(), 'powerlaw', args=(alpha, xmin))
				print(ks_score)
				print(p)
			else:
				return
		elif measure_num == 2:
			# measure_group = self.data.groupby(self.breakdown_fields[0])
			a_series = self.data[self.measure_fields[0]]
			b_series = self.data[self.measure_fields[1]]
			self.breakdown_features = get_statistical_pairwise_features(a_series, b_series)
		else: return
		return self.breakdown_features


if __name__ == '__main__':
	traverse_all()