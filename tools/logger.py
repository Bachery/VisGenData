import logging
import platform

class logger():
	def __init__(self, log_file, parameters):
		self.info_logger = logging.getLogger()
		self.info_logger.setLevel(logging.INFO)
		formatter = logging.Formatter('%(asctime)s - %(message)s')
		file_handler = logging.FileHandler(log_file)
		file_handler.setLevel(logging.INFO)
		file_handler.setFormatter(formatter)
		self.info_logger.addHandler(file_handler)
		self.log(platform.node())
		self.log(platform.platform())
		self.log('TASK INFO:')
		self.log_dict(parameters)

	def log(self, info):
		self.info_logger.info(info)
		print(info)

	def log_dict(self, dict):
		for k, v in dict.items():
			if type(k) != str: k = str(k)
			if type(v) != str: v = str(v)
			self.log(k + ': ' + v)
		# for kv in dict.items():
		# 	self.log(kv)
		# self.log('\n')
