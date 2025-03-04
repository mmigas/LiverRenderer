from math import e


class BloodElement():

    def __init__(self, vf, C_HbT, StO_2, R):
        self.vf = vf  # blood volume fraction
        # self.C_HbT = C_HbT				#total hemoglobin volume fraction in all blood #igual ao vf
        self.StO_2 = StO_2  # liver blood oxigenataion level
        self.R = R  # liver venous mean radius
        self.convertion_factor = 0.0054  # cm-1 - from https://omlc.org/spectra/hemoglobin/
        self.Hb_Data = {}
        self.loadHbTable()

    ##from https://omlc.org/spectra/hemoglobin/
    def loadHbTable(self):
        f = open('liver\\data\\hemoglobin_data.txt', 'r')
        lines = f.readlines()

        for line in lines:
            if line[0] != '#':
                key, hbo2, hb = line.split()
                self.Hb_Data[float(key)] = {"HbO2": float(hbo2), "Hb": float(hb)}
        # print key, '\t', self.Hb_Data[key]

    # absorption coefficients of fully oxygenated blood
    def u_a_HbO2(self, y):
        u_a = self.interpolateTable(y, "HbO2")
        return u_a * self.convertion_factor

    # absorption coefficients of fully deoxygenated blood	
    def u_a_Hb(self, y):
        u_a = self.interpolateTable(y, "Hb")
        return u_a * self.convertion_factor

    # absorption coefficients of fully oxygenated and deoxygenated whole blood
    # given an average hemoglobin concentration of 150 mg/ml.
    def u_a_HbT(self, y):
        return self.StO_2 * self.u_a_HbO2(y) + (1.0 - self.StO_2) * self.u_a_Hb(y)

    # pigment packaging factor
    def C(self, y):
        return (1.0 - e ** (-2.0 * self.R * self.u_a_HbT(y))) / (2.0 * self.R * self.u_a_HbT(y))

    # blood absorbance coefficient modelling in cm-1
    def u_a(self, y):
        y = str(y)
        return self.C(y) * self.vf * self.u_a_HbT(y)  # * blood_vf #cm-1

    def interpolateTable(self, target, type):
        # print 'target: ', target

        if (float(target) in self.Hb_Data): return self.Hb_Data[float(target)][type]

        last_key = last_value = 0

        for key in sorted(self.Hb_Data):
            # print 'key: ', key, ' data: ', self.Hb_Data[key][type]
            # os.system('pause')
            if (float(key) > float(target)):
                return self.lerp(last_key, key, float(target), float(last_value), float(self.Hb_Data[key][type]))

            last_key = key
            last_value = self.Hb_Data[float(key)][type]

            pass

        pass

    def lerp(self, lower, upper, target, data_lower, data_upper):
        d1 = float(upper) - float(lower)
        d2 = float(target) - float(lower)
        time = d2 / d1

        # print 'teste: ', lower,' ',  upper,' ', target,' ', data_lower,' ', data_upper

        return (1.0 - time) * float(data_lower) + time * float(data_upper)


# testes
def calculate_absorption(vf, C_HbT, StO_2, R, y):
    return(BloodElement(float(vf), float(C_HbT), float(StO_2), float(R))).u_a(y)

    