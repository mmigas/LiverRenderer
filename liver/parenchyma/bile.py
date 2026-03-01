import os


class BileElement():

    def __init__(self, vf):
        self.vf = vf  # bile volume fraction at liver
        self.bile_Data = {}
        self.loadBileTable()

    # from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3047365/
    def loadBileTable(self):
        # Get the path of the bile_data.txt file through the parent directory of this script
        bile_data_file = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "\\data\\bile_data.txt"
        f = open(bile_data_file, 'r')
        lines = f.readlines()

        for line in lines:
            if line[0] != '#':
                key, u_a = line.split()
                self.bile_Data[float(key)] = u_a

    # print key, '\t', self.bile_Data[key]

    def lerp(self, lower, upper, target, data_lower, data_upper):
        d1 = float(upper) - float(lower)
        d2 = float(target) - float(lower)
        time = d2 / d1

        # print lower,' ',  upper,' ', target,' ', data_lower,' ', data_upper

        return (1.0 - time) * float(data_lower) + time * float(data_upper)

    def interpolateTable(self, target):
        if (float(target) in self.bile_Data): return self.bile_Data[target]

        last_key = last_value = 0

        for key in sorted(self.bile_Data):
            # print key, ' ', H20_Data[key]

            if (float(key) > float(target)):
                return self.lerp(last_key, key, target, last_value, self.bile_Data[key])

            last_key = key
            last_value = self.bile_Data[key]

            pass

        pass

    def u_a(self, target):
        return round(float(self.interpolateTable(target)) * self.vf, 6)


def calculate_absorption(vf, y):
    return (BileElement(float(vf))).u_a(int(y))
    pass