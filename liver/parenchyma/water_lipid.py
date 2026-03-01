from math import e
import numpy as np
from water import *
from lipid import *
import sys


class WaterLipidElement():

    def __init__(self, water_vf, lipid_vf):
        self.water_vf = water_vf
        self.lipid_vf = lipid_vf
        self.vWL = (lipid_vf * water_vf) + water_vf  # 0.91 #from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3047365/
        self.Water = WaterElement()
        self.Lipid = LipidElement()

    def u_a(self, target):
        return self.vWL * (self.lipid_vf * self.Lipid.u_a(target) + (1.0 - self.lipid_vf) * self.Water.u_a(target))
        pass


def calculate_absorption(water_vf, lipid_vf, y):
    wl = WaterLipidElement(water_vf, lipid_vf)
    return wl.u_a(y)
    pass
