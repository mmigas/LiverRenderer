import imageio
import numpy as np
import OpenEXR
import Imath
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim


def load_exr_image(file_path):
    """
    Loads an EXR image with HALF precision channels and returns a NumPy array in float32.
    Assumes the image has 'R', 'G', and 'B' channels.
    """
    exr_file = OpenEXR.InputFile(file_path)
    header = exr_file.header()

    # Determine image dimensions from the data window
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    # Use the HALF pixel type since channels are stored as HALF.
    # (This tells OpenEXR to provide the raw half data.)
    pt = Imath.PixelType(Imath.PixelType.HALF)

    # Specify the channel order. (Adjust as needed if your channels are different.)
    channels = ['R', 'G', 'B']
    channel_arrays = []

    for ch in channels:
        if ch not in header['channels']:
            raise ValueError(f"Channel '{ch}' not found in EXR file.")
        # Get raw bytes for the channel. 
        # For safety, force a copy by wrapping the result in bytes().
        raw_bytes = bytes(exr_file.channel(ch, pt))
        # The raw data is in HALF format: 16 bits per value.
        # First, read as unsigned 16-bit integers.
        arr_uint16 = np.frombuffer(raw_bytes, dtype=np.uint16)
        # Interpret the uint16 data as float16 values.
        arr_float16 = arr_uint16.view(np.float16)
        # Convert to float32 for further processing.
        arr_float32 = arr_float16.astype(np.float32)

        if arr_float32.size != width * height:
            raise ValueError(f"Channel '{ch}' has unexpected size: {arr_float32.size} vs {width * height}")
        # Reshape to the image dimensions.
        channel_arrays.append(arr_float32.reshape((height, width)))

    # Stack channels into a (H, W, 3) image
    img = np.stack(channel_arrays, axis=-1)
    return img


def load_image(file_path):
    """Loads PNG or EXR images into NumPy arrays."""
    if file_path.lower().endswith(".exr"):
        return load_exr_image(file_path)
    return imageio.imread(file_path).astype(np.float32)


def calculate_mse(reference, rendered):
    err = np.sum((rendered.astype(np.float32) - reference.astype(np.float32)) ** 2)

    # Normalize by the total number of pixels
    err /= float(rendered.shape[0] * reference.shape[1])

    return err


def calculate_ssim(reference, rendered):
    """
    Computes the Structural Similarity Index (SSIM) between two images.
    SSIM is computed per channel and averaged.
    Returns the average SSIM and the SSIM map.
    """
    ssim_values = []
    ssim_maps = []
    for i in range(reference.shape[-1]):  # Loop through R, G, B channels
        ssim_channel, ssim_map = ssim(reference[..., i], rendered[..., i], data_range=reference.max() - reference.min(), full=True)
        ssim_values.append(ssim_channel)
        ssim_maps.append(ssim_map)

    avg_ssim = np.mean(ssim_values)  # Average SSIM across channels
    ssim_map = np.mean(ssim_maps, axis=0)  # Average SSIM map across channels

    return avg_ssim, ssim_map

def visualize_ssim(ssim_map):
    """
    Visualizes the SSIM map using matplotlib.
    """
    plt.imshow(ssim_map, cmap='viridis')
    plt.colorbar()
    plt.title('SSIM Map')
    plt.show()

def compare_images(reference_path, rendered_path):
    """
    Loads two images and calculates MSE & SSIM.
    """
    reference = load_image(reference_path)
    rendered = load_image(rendered_path)

    if reference.shape != rendered.shape:
        raise ValueError("Images must have the same dimensions and channels.")

    mse_value = calculate_mse(reference, rendered)
    ssim_value, ssim_map = calculate_ssim(reference, rendered)

    return mse_value, ssim_value, ssim_map


def main():
    image_path1 = 'C:\dev\LiverRenderer\scenes\Liver\mitsuba0.6\outputs\liver - 700.exr'
    image_path2 = 'C:\dev\LiverRenderer\scenes\Liver\mitsuba3\outputs\liver - 700.exr'
    mse, ssim, ssim_map = compare_images(image_path1, image_path2)
    print(f"MSE: {mse:.4f}")
    print(f"SSIM: {ssim:.4f}")
    visualize_ssim(ssim_map)

if __name__ == "__main__":
    OPENCV_IO_ENABLE_OPENEXR = True
    main()
