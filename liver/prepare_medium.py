import importlib
import os
import sys

# Define the constants map with arguments for each script
ENZYME_MAP = {
    "collagen1": ["collagen_d", "collagen_n_med", "collagen_n_p", "collagen_vf1"],
    "collagen2": ["collagen_d", "collagen_n_med", "collagen_n_p", "collagen_vf2"],
    "collagen3": ["collagen_d", "collagen_n_med", "collagen_n_p", "collagen_vf3"],
    "collagen4": ["collagen_d", "collagen_n_med", "collagen_n_p", "collagen_vf4"],
    "elastin1": ["elastin_d", "elastin_n_med", "elastin_n_p", "elastin_vf1"],
    "elastin2": ["elastin_d", "elastin_n_med", "elastin_n_p", "elastin_vf2"],
    "elastin3": ["elastin_d", "elastin_n_med", "elastin_n_p", "elastin_vf3"],
    "elastin4": ["elastin_d", "elastin_n_med", "elastin_n_p", "elastin_vf4"],

    "blood": ["blood_vf", "blood_C_HbT", "blood_St0_2", "blood_R"],
    "bile": ["bile_vf"],
    "water_lipid": ["water_lipid_water_vf", "water_lipid_lipid_vf"],
    "hepatocity": ["hepatocity_vf", "hepatocity_l_axis", "hepatocity_g_axis"],
}
# Replace these with actual argument values
ARGUMENT_VALUES = {
    "blood_vf": 0.002, "blood_C_HbT": 0.0177, "blood_St0_2": 0.9084, "blood_R": 0.004,
    "bile_vf": 0.0005,
    "water_lipid_water_vf": 0.7, "water_lipid_lipid_vf": 0.019,
    "hepatocity_vf": 0.8, "hepatocity_l_axis": 0.0020, "hepatocity_g_axis": 0.0030,

    "collagen_d": 3.5, "collagen_n_med": 1.35, "collagen_n_p": 1.5,
    "collagen_vf1": 0.949,
    "collagen_vf2": 0.810,
    "collagen_vf3": 0.001,
    "collagen_vf4": 0.007,
    "elastin_d": 0.5, "elastin_n_med": 1.33, "elastin_n_p": 1.534,
    "elastin_vf1": 0.051,
    "elastin_vf2": 0.189,
    "elastin_vf3": 0.254,
    "elastin_vf4": 0.087,
}


def run_script(module_name, wavelength):
    """
    Runs a given script with its arguments and returns its output as a float.

    Parameters:
    - script_name (str): The name of the script to run.
    - wavelength (int): The wavelength to pass as the last argument.

    Returns:
    - float: The output of the script.
    """
    args = [ARGUMENT_VALUES[arg_name] for arg_name in ENZYME_MAP[module_name]]
    try:
        # Dynamically import the module
        if module_name == "collagen1" or module_name == "collagen2" or module_name == "collagen3" or module_name == "collagen4":
            module_name = "collagen"
        elif module_name == "elastin1" or module_name == "elastin2" or module_name == "elastin3" or module_name == "elastin4":
            module_name = "elastin"

        module = importlib.import_module(module_name)
        # Call the calculate_absorption function from the module
        if module_name == "hepatocity":
            return module.calculate_absorption(*args)
        else:
            return module.calculate_absorption(*args, wavelength)
    except AttributeError:
        print(f"Error: {module_name} does not have a `calculate_absorption` function.")
        return 0.0
    except ModuleNotFoundError:
        print(f"Error: Module {module_name} not found.")
        return 0.0


def calc_abs_coeff(start_wavelength, end_wavelength, step, spectrum):
    if (spectrum == "hepatocity"):
        return f"{run_script(spectrum, start_wavelength)}"  # hepatocity does not depend on wavelength
    else:
        red_channel = green_channel = blue_channel = 0
        red_steps = green_steps = blue_steps = 0

        for wavelength in range(start_wavelength, end_wavelength + 1, step):
            sigma = run_script(spectrum, wavelength)
            if 680 < wavelength < 720:
                red_channel += sigma
                red_steps += 1
            elif 520 < wavelength < 570:
                green_channel += sigma
                green_steps += 1
            elif 410 < wavelength < 460:
                blue_channel += sigma
                blue_steps += 1

        red_channel /= red_steps
        green_channel /= green_steps
        blue_channel /= blue_steps
        if ("collagen" in spectrum or "elastin" in spectrum):
            return f"{red_channel}", f"{green_channel}", f"{blue_channel}"
        else:
            return f"{red_channel}, {green_channel}, {blue_channel}"


# Example usage
if __name__ == "__main__":
    start_wavelength = 360  # Replace with the starting wavelength
    end_wavelength = 710  # Replace with the ending wavelength
    step = 1  # Step size for wavelengths
    temp = 269.26180490217416
    # Get the folder in which this script is located
    parenchyma_folder = os.path.dirname(os.path.abspath(__file__)) + "\\parenchyma"
    sys.path.append(parenchyma_folder)
    glisson_folder = os.path.dirname(os.path.abspath(__file__)) + "\\glisson"
    sys.path.append(glisson_folder)
    absorption_results = calc_abs_coeff(start_wavelength, end_wavelength, step, "collagen1")
    print(absorption_results)
