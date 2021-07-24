from .TimelineError import TimelineErrors

class Data:
	def __init__(
		self, 
		hasTrafficLight, 
		TrafficLightState, 
		hasLicensePlate, 
		licensePlatePosition, 
		vehicles
	):
		self.hasTrafficLight = hasTrafficLight
		self.TrafficLightState = TrafficLightState
		self.hasLicensePlate = hasLicensePlate
		self.licensePlatePosition = licensePlatePosition
		self.vehicles = vehicles
		

class VehicleData:
	def __init__(
		self, 
		codename,
		position,
		possiblePositionAtTheNextMoment
	):
		self.codename = codename
		self.position = position
		self.possiblePositionAtTheNextMoment = possiblePositionAtTheNextMoment


class Timeline:
	def __init__(self):
		self.timestamps = []

	def stamp(self, time):
		if not isinstance(time, time):
			raise TimelineErrors.ArgumentTypeError(time, time)
		self.timestamps.append(time)
		return self
