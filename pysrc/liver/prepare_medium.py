import importlib
import os
import sys

# Define the constants map with arguments for each script
CONSTANTS_MAP = {
    #"blood": ["blood_vf", "blood_C_HbT", "blood_St0_2", "blood_R"],
    #"bile": ["bile_vf"],
    #    "hepatocity.py": ["hepatocity_args1", "hepatocity_args2", "hepatocity_args3", "hepatocity_args4"],
    #    "lipid.py": ["lipid_args1", "lipid_args2", "lipid_args3", "lipid_args4"],
    "water_lipid": ["water_lipid_water_vf", "water_lipid_lipid_vf"],
}
# Replace these with actual argument values
ARGUMENT_VALUES = {
    #"blood_vf": 0.002, "blood_C_HbT": 0.0177, "blood_St0_2": 0.9084, "blood_R": 0.004,
    #"bile_vf": 0.0005,
    # "hepatocity_args1": 1.2, "hepatocity_args2": 2.2, "hepatocity_args3": 3.2, "hepatocity_args4": 4.2,
    "water_lipid_water_vf": 0.7, "water_lipid_lipid_vf": 0.019,
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
    args = [ARGUMENT_VALUES[arg_name] for arg_name in CONSTANTS_MAP[module_name]]
    try:
        # Dynamically import the module
        module = importlib.import_module(module_name)
        # Call the calculate_absorption function from the module
        return module.calculate_absorption(*args, wavelength)
    except AttributeError:
        print(f"Error: {module_name} does not have a `calculate_absorption` function.")
        return 0.0
    except ModuleNotFoundError:
        print(f"Error: Module {module_name} not found.")
        return 0.0


def calculate_absorption_coefficients(start_wavelength, end_wavelength, step):
    """
    Runs all scripts for each wavelength and calculates the sum of their results.

    Parameters:
    - start_wavelength (int): Starting wavelength.
    - end_wavelength (int): Ending wavelength.
    - step (int): Step size for wavelengths.

    Returns:
    - dict: A dictionary with wavelengths as keys and the sum of the results as values.
    """
    scripts = list(CONSTANTS_MAP.keys())
    absorption_map = {}

    for wavelength in range(start_wavelength, end_wavelength + 1, step):
        total_absorption = 0.0
        for module in scripts:
            total_absorption += run_script(module, wavelength)
            print(f"{module} at {wavelength}nm: {total_absorption:.4f}")
        absorption_map[wavelength] = total_absorption

    return absorption_map


# Example usage
if __name__ == "__main__":

    start_wavelength = 360  # Replace with the starting wavelength
    end_wavelength = 710  # Replace with the ending wavelength
    step = 10  # Step size for wavelengths
    # Get the folder in which this script is located
    parenchyma_folder = os.path.dirname(os.path.abspath(__file__)) + "\\parenchyma"
    sys.path.append(parenchyma_folder)
    absorption_results = calculate_absorption_coefficients(start_wavelength, end_wavelength, step)
    # Print final absorption map
    print("\nFinal Absorption Map:")
    for wavelength, total in absorption_results.items():
        # Print the wavelength and total absorption with 0.4f precision
        print(f"{wavelength}:{total + 269.26180490217416:.4f},")