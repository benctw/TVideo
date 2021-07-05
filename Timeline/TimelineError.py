

class TimelineError(Exception):
	pass


class TimelineErrors:

	@staticmethod
	def ArgumentTypeError(arg):
		return TimelineError(arg, 'The argument must be type of Time!')