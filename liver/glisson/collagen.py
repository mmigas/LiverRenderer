import numpy as np
import miepython as mp

class CollagenElement:
        
    def compute_mie(self, fv, a, y, n_med, n_p):
        # n_med = 1.33  # medium refractive index
        # n_p = 1.5  # particle refractive index
        # a = 3.500  #microns - particle radius
        # y = 0.6328  #microns  - wavelength in vacuo
        # fv =  0.0019 # volume fraction of particles in medium
    
        m = n_p / n_med
        x = 2.0 * np.pi * a / (y / n_med)
        A = np.pi * (a ** 2)
        # ps = fv/((4.0/3.0)*np.pi*(a**3))  #um-3  #as a sphere
        ps = fv / ((np.pi * ((a * 2) ** 2)) / 4.0)  # as a cylinder (Jacques 1996)	
    
        qext, qsca, qback, g = mp.mie(m, x)
    
        sigma_s_m = qsca * A
        micro_s_m = ps * sigma_s_m  # * 10**2
    
        return micro_s_m  # Mie Coefficient (cm-1)


# teste
def calculate_absorption(d, n_med, n_p, vf, y):
    return (CollagenElement().compute_mie(float(vf), float(d) / 2.0, float(y) / 1000.0, float(n_med), float(n_p)))