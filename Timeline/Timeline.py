from .TimelineError import TimelineErrors

class Timeline:

	def __init__(self):
		self.timestamps = []

	def stamp(self, time):
		if not isinstance(time, time):
			raise TimelineErrors.ArgumentTypeError(time)
		self.timestamps.append(time)
