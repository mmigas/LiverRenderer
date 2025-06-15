import mitsuba as mi
import os

# --- Configuration ---
# 1. Set the Mitsuba variant (must be CUDA-enabled)
try:
    mi.set_variant('cuda_ad_rgb')
except ImportError:
    print("Error: Could not set the 'cuda_ad_rgb' variant.")
    print("Ensure Mitsuba 3 is compiled with CUDA support and necessary dependencies (PyTorch) are installed.")
    exit(1)

# 2. Define input and output paths
input_exr_path = 'D:\\dev\\LiverRenderer\\scenes\\Liver-SingleMesh\\mitsuba3\\scene.png'
output_denoised_exr_path = "D:\\dev\\LiverRenderer\\scenes\\Liver-SingleMesh\\mitsuba3\\sceneDenoised.exr"
output_denoised_png_path = "D:\\dev\\LiverRenderer\\scenes\\Liver-SingleMesh\\mitsuba3\\sceneDenoised.png" # Optional

# --- Processing ---

# 3. Load the multi-channel EXR
try:
    print(f"Loading EXR: {input_exr_path}")
    input_bitmap = mi.Bitmap(input_exr_path)
    print(f"Loaded bitmap with shape: {input_bitmap.size()} channels: {input_bitmap.channel_count()}")
    #print(f"Available channels: {input_bitmap.metadata().channel_names()}")
except Exception as e:
    print(f"Error loading EXR file: {e}")
    exit(1)

# 4. Extract necessary channels into separate bitmaps
#    IMPORTANT: Channel names depend on how your scene was configured!
#    Check 'Available channels' output above and adjust names if needed.

# 5. Instantiate the OptiX Denoiser
#    Mitsuba's OptixDenoiser figures out input requirements based on the AOVs provided.
#    It usually expects at least color, albedo, and normals.
try:
    print("Initializing OptiX Denoiser...")
    # Let Mitsuba automatically detect inputs based on the dictionary keys
    # The denoiser usually assumes specific keys like 'albedo', 'normals'

    denoiser = mi.OptixDenoiser(input_bitmap.size())
    print("OptiX Denoiser initialized.")
except Exception as e:
    print(f"Error initializing OptiX Denoiser: {e}")
    print("Ensure your GPU drivers and CUDA installation are compatible with OptiX.")
    exit(1)


# 6. Perform Denoising
try:
    print("Denoising...")
    # Pass the noisy color image and the dictionary of AOVs
    denoised_bitmap = denoiser(input_bitmap)
    print("Denoising complete.")
except Exception as e:
    print(f"Error during denoising: {e}")
    exit(1)

# 7. Save the Denoised Output

# --- Save as EXR (Recommended for quality) ---
try:
    print(f"Saving denoised EXR to: {output_denoised_exr_path}")
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_denoised_exr_path), exist_ok=True)
    # Use mi.util.write_bitmap for more control if needed, or bitmap.write()
    denoised_bitmap.write(output_denoised_exr_path)
    print("Denoised EXR saved successfully.")
except Exception as e:
    print(f"Error saving denoised EXR: {e}")

# --- Save as PNG (Optional, involves potential data loss) ---
try:
    print(f"Saving denoised PNG to: {output_denoised_png_path}")
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_denoised_png_path), exist_ok=True)

    # PNG typically requires conversion (e.g., to sRGB, 8-bit).
    # This will clamp HDR values and reduce precision.
    # You might want to apply tone mapping before converting.
    # Example: Simple conversion to sRGB Float, then save (still float PNG if supported)
    # Or convert further to 8-bit for standard PNG.

    # Simple conversion using Mitsuba utilities (adjust as needed)
    png_bitmap = mi.Bitmap(denoised_bitmap, mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8, srgb_gamma=True)

    mi.util.write_bitmap(output_denoised_png_path, png_bitmap)
    print("Denoised PNG saved successfully.")
    print("Note: Saving to PNG involves conversion and potential loss of HDR data.")

except Exception as e:
    print(f"Error saving denoised PNG: {e}")