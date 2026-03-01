import numpy as np
import sys
class ElastingElement():
    def rayleigh(self, m,x):
        # equations 5.7 - 5.9 from Bohren and Huffman
        ratio = (m**2-1)/(m**2+2)
        qsca = 8/3*x**4*abs(ratio)**2
        qext = 4*x*ratio*(1+x**2/15*ratio*(m**4+27*m**2+38)/(2*m**2+3))
        qext = abs(qext.imag + qsca)
        qback = 4*x**4*abs(ratio)**2
        g = 0
        return qext, qsca, qback, g
    
    def compute_rayleigh(self, fv, a, y, n_med, n_p):
        #n_med = 1.33  # medium refractive index
        #n_p = 1.5  # particle refractive index
        #a = 3.500  #microns - particle radius
        #y = 0.6328  #microns  - wavelength in vacuo
        #fv =  0.0019 # volume fraction of particles in medium
            
        m = n_p/n_med
        x = 2.0*np.pi*a/(y/n_med)
        A = np.pi*(a**2.0)
        #ps = fv/((4.0/3.0)*np.pi*(a**3))  #um-3  #as a sphere
        ps = fv/((np.pi*((a*2.0)**2))/4.0)		  #as a cylinder (Jacques 1996)	
    
        rext, rsca, rback, rg = self.rayleigh(m,x)
        
        sigma_s_r = rsca*A
        micro_s_r = ps*sigma_s_r #* 10**2
        
    
        return micro_s_r #Rayleigh Coefficient (cm-1)


#teste
def calculate_absorption(d, n_med, n_p, vf, y):
    return ElastingElement().compute_rayleigh(float(vf), float(d) / 2.0, float(y) / 1000.0, float(n_med), float(n_p))
    
