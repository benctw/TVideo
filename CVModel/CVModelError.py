

class CVModelError(Exception):
    pass

class CVModelErrors():
    pass

class DetectResultError(Exception):
	pass

class DetectResultErrors():

	@staticmethod
	def ArgumentTypeError(*arg, type):
		return DetectResultError(arg, "The argument must be type of {}".format(type))