import mitsuba as mi
import time 
# Set the desired Mitsuba variant
mi.set_variant('cuda_ad_rgb') # Change to 'cuda_ad_rgb' for better performance

# --- Desired rendering parameters ---
output_filename = "bunny" # Updated filename slightly
image_width = 1080
image_height = 1080
integrator_max_depth = -1
samples_per_pixel = 4096

# --- Fog Medium Parameters (using sigma_t and albedo) ---
# sigma_t: Extinction coefficient (how quickly light is attenuated).
#          Higher values mean denser/more opaque fog.
# albedo: Single-scattering albedo (ratio of scattering to extinction).
#         Value between 0 and 1.
#         1.0 means purely scattering (no absorption, e.g., white steam).
#         0.0 means purely absorbing (e.g., black smoke, though this would be dark).
#         Values in between give colored/greyish scattering.

# These values are equivalent to sigma_a=[0.05,0.05,0.05] and sigma_s=[0.15,0.15,0.15]
fog_sigma_t_rgb = [0.20, 0.20, 0.20] # Extinction coefficient (sigma_a + sigma_s)
fog_albedo_rgb  = [0.75, 0.75, 0.75]  # Scattering albedo (sigma_s / (sigma_a + sigma_s))

fog_scale = 2.5 # Overall density scale for the fog. Adjust to make fog thicker/thinner.
# This will scale sigma_t.

# 1. Start building the new scene dictionary
scene_components = {}

# 2. Define the fog medium first using sigma_t and albedo
scene_components['fog_medium_id'] = {
    'type': 'homogeneous',
    'sigma_t': {'type': 'rgb', 'value': fog_sigma_t_rgb}, # Extinction coefficient
    'albedo':  {'type': 'rgb', 'value': fog_albedo_rgb},  # Single-scattering albedo
    'scale': fog_scale,                                  # Scales sigma_t
    'phase': {'type': 'isotropic'}                       # How light scatters
}
print("Fog medium ('fog_medium_id') defined using sigma_t and albedo.")

# 3. Define the volumetric integrator
scene_components['volpath_integrator'] = {
    'type': 'volpath',
    'max_depth': integrator_max_depth
}
print(f"Integrator ('volpath_integrator') defined with max_depth: {integrator_max_depth}.")

# 4. Get the base Cornell Box elements
original_cb_dict = mi.cornell_box()

# 5. Iterate over Cornell Box elements and add them to our scene_components,
#    modifying the sensor and skipping/replacing the original integrator.
sensor_key_in_cb = None

for key, value in original_cb_dict.items():
    if not isinstance(value, dict):
        scene_components[key] = value
        continue

    element_type = value.get('type')

    if element_type in ['perspective', 'thinlens', 'orthogonal', 'camera']: # Heuristic for sensor
        print(f"Found sensor-like element with key '{key}' and type '{element_type}'. Modifying it.")
        sensor_config = value.copy()

        sensor_config.setdefault('film', {'type': 'hdrfilm', 'pixel_format': 'rgb', 'rfilter': {'type': 'gaussian'}})
        sensor_config['film']['width'] = image_width
        sensor_config['film']['height'] = image_height
        print(f"  Updated film resolution to {image_width}x{image_height}.")

        sensor_config['medium'] = {
            'type': 'ref',
            'id': 'fog_medium_id'
        }
        print("  Assigned 'fog_medium_id' as medium.")
        scene_components[key] = sensor_config
        sensor_key_in_cb = key

    elif element_type in ['path', 'direct', 'volpath', 'ao', 'integrator']: # Heuristic for integrator or if key is 'integrator'
        print(f"Found integrator-like element with key '{key}' and type '{element_type}'. It will be replaced by our 'volpath_integrator'.")
        # Skip adding the old integrator to scene_components
        continue

    elif key == 'type' and value == 'scene':
        scene_components[key] = value
    else:
        scene_components[key] = value

if not sensor_key_in_cb:
    print("Warning: Could not automatically find and modify the sensor from mi.cornell_box(). Using default key 'sensor'.")
    # Fallback: if mi.cornell_box() provides a sensor with the key 'sensor' and it wasn't caught above.
    if 'sensor' in original_cb_dict and isinstance(original_cb_dict['sensor'], dict):
        sensor_config = original_cb_dict['sensor'].copy()
        sensor_config.setdefault('film', {'type': 'hdrfilm', 'pixel_format': 'rgb', 'rfilter': {'type': 'gaussian'}})
        sensor_config['film']['width'] = image_width
        sensor_config['film']['height'] = image_height
        sensor_config['medium'] = {'type': 'ref', 'id': 'fog_medium_id'}
        scene_components['sensor'] = sensor_config
        print("  Fallback: Modified sensor with key 'sensor'.")
    else:
        print("Error: Sensor with key 'sensor' not found or not a dict. Please check sensor configuration.")


# Ensure our chosen integrator is used with the standard key 'integrator'.
# Remove any integrator that might have been copied if its key wasn't 'integrator' or caught by type.
keys_to_delete = [k for k,v in scene_components.items() if isinstance(v,dict) and v.get('type') in ['path','direct','ao'] and k != 'volpath_integrator']
for k_del in keys_to_delete:
    del scene_components[k_del]

if 'volpath_integrator' in scene_components :
    scene_components['integrator'] = scene_components.pop('volpath_integrator')
elif 'integrator' not in scene_components : # If no integrator is present at all
    scene_components['integrator'] = {
        'type': 'volpath',
        'max_depth': integrator_max_depth
    }
# If an integrator with key 'integrator' was copied and is not volpath, replace it
elif scene_components['integrator']['type'] != 'volpath':
    print(f"Replacing existing integrator {scene_components['integrator']['type']} with volpath.")
    scene_components['integrator'] = {
        'type': 'volpath',
        'max_depth': integrator_max_depth
    }


print(f"Final integrator type: {scene_components.get('integrator', {}).get('type')}")

# 6. Load the fully constructed scene dictionary
try:
    scene = mi.load_dict(scene_components)
    scene = mi.load_file("D:\\dev\\LiverRenderer\\resources\\data\\docs\scenes\\medium_homogeneous_sss.xml")
    print("Scene loaded successfully.")
except RuntimeError as e:
    print(f"Error loading scene: {e}")
    print("Final scene components submitted to mi.load_dict():")
    import json
    print(json.dumps(scene_components, indent=2, default=lambda o: str(o))) # Print complex objects as strings
    raise # Re-raise the exception

# 7. Render the scene
print(f"Starting rendering with {samples_per_pixel} spp...")
start_time = time.time() # Record start time
image = mi.render(scene, spp=samples_per_pixel)
end_time = time.time()   # Record end time

rendering_duration_seconds = end_time - start_time # Calculate duration in seconds


if rendering_duration_seconds < 60:
    duration_str = f"{rendering_duration_seconds:.2f} seconds"
elif rendering_duration_seconds < 3600:
    minutes = int(rendering_duration_seconds // 60)
    seconds = rendering_duration_seconds % 60
    duration_str = f"{minutes} minute(s) and {seconds:.2f} seconds"
else:
    hours = int(rendering_duration_seconds // 3600)
    minutes = int((rendering_duration_seconds % 3600) // 60)
    seconds = (rendering_duration_seconds % 3600) % 60
    duration_str = f"{hours} hour(s), {minutes} minute(s) and {seconds:.2f} seconds"


# 8. Write the rendered image to a file
output_png = f"{output_filename}.png"
output_exr = f"{output_filename}.exr"
mi.util.write_bitmap(output_png, image)
mi.util.write_bitmap(output_exr, image)

print(f"Rendering complete in {duration_str}. Image saved to {output_png} and {output_exr}")
