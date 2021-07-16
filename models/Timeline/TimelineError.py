

class TimelineError(Exception):
	pass


class TimelineErrors:

	@staticmethod
	def ArgumentTypeError(*arg, type):
		return TimelineError(arg, "The argument must be type of {}!".format(type))