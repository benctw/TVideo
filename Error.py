
class CVModelError(Exception):

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
		return f’{self.message}’ if self.cause is not None else f’{self.cause} : {self.message}’


class TimelineError(Exception):
    pass


class CVModelErrors:

	@staticmethod
	def notLoaded():
		return CVModelError('Model is not loaded!')

	# 定義其他 Model Error
	@staticmethod
	def otherError():
        return CVModelError('Error message!')


class TimelineErrors:

	@staticmethod
	def ArgumentTypeError(arg):
		return TimelineError(arg, 'The argument must be type of Time!')