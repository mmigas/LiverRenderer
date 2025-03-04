from math import e
import numpy as np
import matplotlib.pyplot as plt

Lipid_Data = {}

class LipidElement():

	def __init__(self):
		self.Lipid_Data = {}
		self.loadLipidTable()

	#from https://omlc.org/spectra/fat/fat.txt
	def loadLipidTable(self):
		f = open('liver\\data\\lipid_data.txt', 'r')
		lines = f.readlines()
			
		for line in lines:
			if line[0] != '#':
				key, u_a = line.split()
				self.Lipid_Data[float(key)] = float(u_a) / 100.0			
				#print key, '\t', self.Lipid_Data[key]
			
	def lerp(self, lower, upper, target, data_lower, data_upper):
		d1 = float(upper) - float(lower)
		d2 = float(target) - float(lower)
		time = d2/d1
		
		#print lower,' ',  upper,' ', target,' ', data_lower,' ', data_upper
		
		return (1.0 - time) * float(data_lower) + time * float(data_upper)


	def interpolateTable(self, target):
		if(float(target) in self.Lipid_Data): return self.Lipid_Data[target]
		
		last_key = last_value = 0
		
		for key in sorted(self.Lipid_Data):		
			#print key, ' ', Lipid_Data[key]
			
			if(float(key) > float(target)):
				return self.lerp(last_key, key, target, last_value, self.Lipid_Data[key])
						
			last_key = key
			last_value  = self.Lipid_Data[key]			
			
			pass
		
		pass
			
	def u_a(self, target):
		return round(float(self.interpolateTable(target)), 6)

if __name__ == '__main__':	
	le = LipidElement()
	
	#plot da curva
	x = []
	xl = []
	for i in range(430, 1000, 2):
		l = le.u_a(i)
		x.append(i)		
		xl.append(l)

	plt.plot(x, xl, 'b') 
	plt.yscale('symlog')
	plt.ylabel('absorbance coefficient ($cm^{-1}$)')
	plt.xlabel('wavelength (nm)')
	plt.show()



