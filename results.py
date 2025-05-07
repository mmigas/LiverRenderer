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
    rmse = np.sqrt(err)
    return rmse


def calculate_ssim(reference, rendered):
    """
    Computes the Structural Similarity Index (SSIM) between two images.
    SSIM is computed per channel and averaged.
    Returns the average SSIM and the SSIM map.
    """
    ssim_values = []
    ssim_maps = []
    for i in range(reference.shape[-1]):  # Loop through R, G, B channels
        ssim_channel, ssim_map = ssim(reference[..., i], rendered[..., i], data_range=reference.max() - reference.min(),
                                      full=True)
        ssim_values.append(ssim_channel)
        ssim_maps.append(ssim_map)

    avg_ssim = np.mean(ssim_values)  # Average SSIM across channels
    ssim_map = np.mean(ssim_maps, axis=0)  # Average SSIM map across channels

    return avg_ssim, ssim_map


def visualize_ssim(ssim_map, ssim_value, output_path='results/ssim_map.png'):
    """
    Visualizes the SSIM map using matplotlib and includes the SSIM value in the title.
    """
    plt.imshow(ssim_map, cmap='viridis')
    plt.colorbar()
    plt.axis('off')
    name = output_path.split('/')[-1].split('.')[0]
    name = name.replace('SSIM', '')
    plt.title(f'{name}\'s SSIM Map\n (SSIM: {ssim_value:.4f})')  # Add SSIM value to the title
    plt.savefig(output_path, format='png')  # Save the plot as a PNG file
    plt.show()


def visualize_rmse(reference, rendered, rmse_value, output_path='results/rmse_map.png'):
    """
    Visualizes the pixel-wise squared difference (RMSE map) between two images and includes the RMSE value in the title.
    """
    # Compute the squared difference
    mse_map = (rendered.astype(np.float32) - reference.astype(np.float32)) ** 2

    # Normalize the MSE map for visualization
    mse_map_normalized = mse_map / mse_map.max()

    # Display the RMSE map
    plt.imshow(mse_map_normalized.mean(axis=-1), cmap='viridis')  # Average across channels
    plt.colorbar()
    plt.axis('off')  # Hide axes
    name = output_path.split('/')[-1].split('.')[0]
    name = name.replace('RSME', '')
    plt.title(f'{name}\'s RMSE Map\n (RMSE: {rmse_value:.4f})')  # Add RMSE value to the title
    plt.savefig(output_path, format='png')  # Save the plot as a PNG file
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
    image_path1 = 'D:\\dev\\LiverRenderer\\scenes\\Liver-SingleMesh\\mitsuba3\\scene_white_background.png'
    image_path2 = 'C:\\Users\\Miguel\\Downloads\\Surgery2.0\\Surgery2.0\\denoised.png'
    reference = load_image(image_path1)
    rendered = load_image(image_path2)
    rmse, ssim, ssim_map = compare_images(image_path1, image_path2)
    print(f"MSE: {rmse:.4f}")
    print(f"SSIM: {ssim:.4f}")
    visualize_ssim(ssim_map, ssim, 'results/OptixWhiteBackgroundSSIM.png')
    visualize_rmse(reference, rendered, rmse, 'results/OptixWhiteBackgroundRSME.png')


if __name__ == "__main__":
    OPENCV_IO_ENABLE_OPENEXR = True
    main()
