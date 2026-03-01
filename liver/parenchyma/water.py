from math import e
import numpy as np
import matplotlib.pyplot as plt

class WaterElement():
	
	def __init__(self):
		self.H20_Data = {}
		self.loadH2OTable()
		
	#from https://omlc.org/spectra/water/data/pope97.txt
	def loadH2OTable(self):
		f = open('liver\\data\\water_data.txt', 'r') 
		lines = f.readlines()
			
		for line in lines:
			if line[0] != '#':
				key, u_a = line.split()
				self.H20_Data[float(key)] = u_a			
				#print key, '\t', self.H20_Data[key]
			
	def lerp(self, lower, upper, target, data_lower, data_upper):
		d1 = float(upper) - float(lower)
		d2 = float(target) - float(lower)
		time = d2/d1
		
		#print lower,' ',  upper,' ', target,' ', data_lower,' ', data_upper
		
		return (1.0 - time) * float(data_lower) + time * float(data_upper)

	def interpolateTable(self, target):
		if(float(target) in self.H20_Data): return self.H20_Data[target]
		
		last_key = last_value = 0
		
		for key in sorted(self.H20_Data):		
			#print key, ' ', H20_Data[key]
			
			if(float(key) > float(target)):
				return self.lerp(last_key, key, target, last_value, self.H20_Data[key])
						
			last_key = key
			last_value  = self.H20_Data[key]			
			
			pass
		
		pass
			
	def u_a(self, target):
		return round(float(self.interpolateTable(target)), 6)

if __name__ == '__main__':	
	we = WaterElement()
	b = we.u_a(440)
	g = we.u_a(510)
	r = we.u_a(650)
	print('(r, g, b) = (', r, ',', g, ',', b, ')')

	#plot da curva
	x = []
	xl = []
	for i in range(380, 700, 2):
		l = we.u_a(i)
		x.append(i)		
		xl.append(l)

	plt.plot(x, xl, 'b')
	plt.yscale('symlog')
	plt.ylabel('absorbance coefficient ($cm^{-1}$)')
	plt.xlabel('wavelength (nm)')	
	plt.show()



