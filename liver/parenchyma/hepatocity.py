from math import*
import sys

#from Chen 2015
#considerando como um esferoide

class HepatocityElement():

	def __init__(self, vf, l_axis, g_axis):
		self.vf = vf
		self.l_axis = l_axis
		self.g_axis = g_axis
	
	def u_g(self):			
		a = self.l_axis
		b = self.g_axis
		c = sqrt(1.0 - ((a*a)/(b*b)))
		S_V = (3.0/(2.0*a)) * (a/b + asin(c)/c) 
		v = self.vf

		return (S_V) * (v/4.0)

def calculate_absorption(vf, l_axis, g_axis):
	return (HepatocityElement(float(vf), float(l_axis), float(g_axis)).u_g())
