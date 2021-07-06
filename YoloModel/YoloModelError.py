

class YoloModelError(Exception):

	def __init__(self, cause, message):
		if message is None:
			self.message = cause
			self.cause = None
			super().__init__(self.message)
		else:
			self.cause = cause
			self.message = message
			super().__init__(cause, message)

	def __str__(self):
		return "{0.message}".format(self) if self.cause is not None else "{0.cause} : {0.message}".format(self)


class YoloModelErrors:

	@staticmethod
	def notLoaded():
		return YoloModelError('Model is not loaded!')

	# 定義其他 Model Error
	@staticmethod
	def otherError():
		return YoloModelError('Error message!')


class DetectResultError(Exception):
	pass


class DetectResultErrors():

	@staticmethod
	def ArgumentTypeError(*arg, type):
		return DetectResultError(arg, "The argument must be type of {}".format(type))