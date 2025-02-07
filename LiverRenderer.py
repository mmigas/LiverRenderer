import importlib
import os
import shutil
import subprocess
import sys
import yaml
import xml.etree.ElementTree as ET



def load_settings():
    try:
        with open('RendererSettings.yml', 'r') as file:
            settings = yaml.safe_load(file)
    except FileNotFoundError:
        # Stop the program if the settings file is not found and print message
        print("Settings file not found.")
        sys.exit(1)
    return settings


def parse_settings(settings):
    model = settings.get('Model')
    if model is None:
        print("Model not found in settings.")
        sys.exit(1)

    if model.lower() == "ref":
        mitsuba_version = 3
    else:
        mitsuba_version = 0.6

    variant = settings.get('Variant')

    if variant is None:
        print("Variant not found in settings. Using default scalar_rgb")
        variant = "scalar_rgb"
    elif variant.lower() != "scalar_rgb" and variant.lower() != "cuda_rgb":
        print("Invalid variant. Using default scalar_rgb")
        variant = "scalar_rgb"

    scene = settings.get('Scene')
    scene = scene.lower()
    if scene is None:
        print("Scene not found in settings.")
        sys.exit(1)
    if not os.path.exists(f"scenes/{scene}/mitsuba{mitsuba_version}/scene.xml"):
        print(f"Scene file scenes/{scene}/mitsuba{mitsuba_version}/scene.xml not found.")
        sys.exit(1)
    scene_folder = f"scenes/{scene}/mitsuba{mitsuba_version}"

    width = settings.get('Resolution').get('Width')
    height = settings.get('Resolution').get('Height')
    if width is None or height is None:
        print("Width or height not found in settings.")
        sys.exit(1)

    spp = settings.get('Samples Per Pixel')
    if spp is None:
        print("Samples Per Pixel not found in settings.")
        sys.exit(1)

    max_depth = settings.get('Max Depth')
    if max_depth is None:
        print("Max Depth not found in settings.")
        sys.exit(1)

    glisson_params = settings.get('Glisson Capsule')
    if glisson_params is None:
        print("Glisson Capsule parameters not found in settings.")
        sys.exit(1)
    parenchyma_params = settings.get('Parenchyma')
    if parenchyma_params is None:
        print("Parenchyma parameters not found in settings.")
        sys.exit(1)

    return model, mitsuba_version, variant, scene, scene_folder, width, height, spp, max_depth, glisson_params, parenchyma_params


def edit_scene(scene_path, model, width, height, spp, max_depth, glisson_params, parenchyma_params):
    temp_path = f"{scene_path}/scene_temp.xml"
    # Copy scene path to a temporary file
    shutil.copy(f"{scene_path}/scene.xml", temp_path)

    tree = ET.parse(f"{scene_path}/scene_temp.xml")
    root = tree.getroot()

    start_wavelength = 360  # Replace with the starting wavelength
    end_wavelength = 710  # Replace with the ending wavelength
    step = 10  # Step size for wavelengths
    blood_spectrum = importlib.import_module("prepare_medium").calculate_absorption_coefficients(start_wavelength, end_wavelength, step, "blood")
    bile_spectrum = importlib.import_module("prepare_medium").calculate_absorption_coefficients(start_wavelength, end_wavelength, step, "bile")
    lipid_water_spectrum = importlib.import_module("prepare_medium").calculate_absorption_coefficients(start_wavelength, end_wavelength, step, "water_lipid")
    hepaticity_spectrum = importlib.import_module("prepare_medium").calculate_absorption_coefficients(start_wavelength, end_wavelength, step, "hepatocity")
    for element in root.findall("default"):
        if element.get("name") == "res_width":
            element.set("value", str(width))
        if element.get("name") == "res_height":
            element.set("value", str(height))
        if element.get("name") == "spp":
            element.set("value", str(spp))
        if element.get("name") == "max_depth":
            element.set("value", str(max_depth))

    if model.lower() == "ref":
        for element in root.findall("medium"):
            if element.get("id") == "parenchymaMedium":
                for child in element:
                    if child.get("name") == "sigma_blood":
                        child.set("value", blood_spectrum)
                    if child.get("name") == "sigma_bile":
                        child.set("value", bile_spectrum)
                    if child.get("name") == "sigma_lipid_water":
                        child.set("value", lipid_water_spectrum)
                    if child.get("name") == "sigma_hepatocity":
                        child.set("value", hepaticity_spectrum)
    elif model.lower() == "ref0.6":
        for element in root.findall("medium"):
            if element.get("id") == "layer2m":
                for child in element:
                    if child.get("name") == "blood_vf":
                        child.set("value", str(parenchyma_params.get("blood_vf")))
                    if child.get("name") == "blood_StO2":
                        child.set("value", str(parenchyma_params.get("blood_St02")))
                    if child.get("name") == "blood_R":
                        child.set("value", str(parenchyma_params.get("blood_r")))
                    if child.get("name") == "bile_vf":
                        child.set("value", str(parenchyma_params.get("bile_vf")))
                    if child.get("name") == "lipid_vf":
                        child.set("value", str(parenchyma_params.get("lipid_vf")))
                    if child.get("name") == "water_vf":
                        child.set("value", str(parenchyma_params.get("water_vf")))
                    if child.get("name") == "hepatocity_vf":
                        child.set("value", str(parenchyma_params.get("hepatocity_vf")))
                    if child.get("name") == "hepatocity_lAxis":
                        child.set("value", str(parenchyma_params.get("hepatocity_lAxis")))
                    if child.get("name") == "hepatocity_gAxis":
                        child.set("value", str(parenchyma_params.get("hepatocity_gAxis")))
    tree.write(temp_path)
    return temp_path


def run_mitsuba(mitsuba_version, variant, scene_folder, scene_name, temp_scene):
    # Get the absolute path of the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to the Mitsuba executable relative to this script
    if mitsuba_version == 0.6:
        try:
            # command = "export MITSUBA_PYVER=3.6.9 && source ../my-mitsuba-master/setpath.sh && mitsuba  -o " + scene_folder + "/" + scene_name + " " + temp_scene
            command = f"export MITSUBA_PYVER=3.6.9 && source ../my-mitsuba-master/setpath.sh && mitsuba -o {scene_folder}/{scene_name} {temp_scene}"

            process = subprocess.Popen(
                ["wsl", "bash", "-c", command],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line-buffered output
                universal_newlines=True
            )

            # Print the output in real-time
            for line in process.stdout:
                print(line, end="")  # Avoids double newlines

            for line in process.stderr:
                print(line, end="")  # Prints errors in real-time

            process.wait()  # Ensure the process completes
        except subprocess.CalledProcessError as e:
            print("An error occurred while running the command:")
            print(e)
            print("Standard Error:")
            print(e.stderr)
    else:
        # Verify that the executable exists
        mitsuba_exe_path = script_dir + f"\\build\\Release\\mitsuba.exe"
        if not os.path.exists(mitsuba_exe_path):
            raise FileNotFoundError(f"Mitsuba executable not found at: {mitsuba_exe_path}")

        # Build the command as a list. Adjust the command-line arguments as needed.
        cmd = [mitsuba_exe_path, "-m" + variant, "-o" + scene_folder + "/" + scene_name, temp_scene]
        #        try:
        #            result = subprocess.run(
        #                cmd,
        #                capture_output=True,  # Capture standard output and error
        #                text=True,  # Decode bytes to string
        #                check=True  # Raise an exception if the command fails
        #            )
        #            print("Mitsuba output:")
        #            print(result.stdout)
        #        except subprocess.CalledProcessError as e:
        #            print("An error occurred while running Mitsuba:")
        #            print(e)
        #            print("Standard Error:")
        #            print(e.stderr)
        # Start the process
        with subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,  # Capture standard output
                stderr=subprocess.STDOUT,  # Merge stderr with stdout
                text=True,  # Return strings instead of bytes (Python 3.7+)
                bufsize=1  # Line-buffered
        ) as process:
            # Read output line by line as it is produced
            for line in process.stdout:
                print(line, end='')  # Already includes newline, so no need for extra '\n'

        # Optionally, check the process's return code
        if process.returncode != 0:
            print(f"\nProcess exited with code: {process.returncode}")

    return f"{scene_folder}\\{scene_name}.exr"


def open_image(image):
    # Open the image with the default program
    subprocess.Popen(["tev", image], creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)


def main():
    liver_folder = os.path.dirname(os.path.abspath(__file__)) + "\\liver"
    sys.path.append(liver_folder)
    parenchyma_folder = os.path.dirname(os.path.abspath(__file__)) + "\\liver\\parenchyma"
    sys.path.append(parenchyma_folder)
    settings = load_settings()
    model, mitsuba_version, variant, scene, scene_folder, width, height, spp, max_depth, glisson_params, parenchyma_params = parse_settings(settings)
    temp_scene = edit_scene(scene_folder, model, width, height, spp, max_depth, glisson_params, parenchyma_params)
    image = run_mitsuba(mitsuba_version, variant, scene_folder, scene, temp_scene)
    open_image(image)
    pass


if __name__ == "__main__":
    main()
