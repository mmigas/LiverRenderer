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

def get_black_mask(mask_path):
    """Returns a boolean mask where True means the pixel is black in the mask image."""
    mask_img = load_image(mask_path)
    # Consider a pixel black if all channels are zero (or very close to zero)
    black_mask = np.all(mask_img < 1e-5, axis=-1)
    return black_mask

def calculate_mse(reference, rendered, mask):
    # Only compare masked pixels
    diff = (rendered.astype(np.float32) - reference.astype(np.float32)) ** 2
    masked_diff = diff[mask]
    mse = np.mean(masked_diff)
    rmse = np.sqrt(mse)
    return rmse


def calculate_ssim(reference, rendered, mask):
    ssim_values = []
    ssim_maps = []
    for i in range(reference.shape[-1]):
        # For SSIM, extract only the masked pixels as 1D arrays
        ssim_channel, ssim_map = ssim(
            reference[..., i][mask], rendered[..., i][mask],
            data_range=1.0,
            full=True
        )
        # Create a full-size map with NaN outside the mask
        full_map = np.full(mask.shape, np.nan)
        full_map[mask] = ssim_map
        ssim_values.append(ssim_channel)
        ssim_maps.append(full_map)
    avg_ssim = np.mean(ssim_values)
    ssim_map = np.nanmean(np.stack(ssim_maps, axis=-1), axis=-1)
    return avg_ssim, ssim_map

def visualize_ssim(ssim_map, ssim_value, mask, output_path='results/ssim_map.png'):
    plt.imshow(np.where(mask, ssim_map, np.nan), cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(label="Per-Pixel SSIM")  # Add a label for clarity
    plt.axis('off')
    name = output_path.split('/')[-1].split('.')[0]
    name = name.replace('SSIM', '')
    plt.title(f'{name}\'s SSIM Map\n (SSIM: {ssim_value:.4f})')
    plt.savefig(output_path, format='png')
    plt.show()

def visualize_rmse(reference, rendered, rmse_value, mask, output_path='results/rmse_map.png'):
    per_pixel_mse = np.mean((rendered.astype(np.float32) - reference.astype(np.float32)) ** 2, axis=-1)
    per_pixel_rmse = np.sqrt(per_pixel_mse)
    per_pixel_rmse_masked = np.where(mask, per_pixel_rmse, np.nan)

    plt.imshow(per_pixel_rmse_masked, cmap='viridis')

    plt.colorbar(label="Per-Pixel RMSE")  # Add a label for clarity
    plt.axis('off')
    name = output_path.split('/')[-1].split('.')[0].replace('RSME', '')

    # Use the corrected overall RMSE value in the title
    plt.title(f'{name}\'s RMSE Map\n (RMSE: {rmse_value:.4f})')

    plt.savefig(output_path, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.show()


def compare_images(reference_path, rendered_path, mask_path):
    reference = load_image(reference_path)
    rendered = load_image(rendered_path)
    mask = get_black_mask(mask_path)
    if reference.shape != rendered.shape or reference.shape[:2] != mask.shape:
        raise ValueError("Images and mask must have the same dimensions.")
    mse_value = calculate_mse(reference, rendered, mask)
    max_val = reference.max()
    if max_val > 0:
        reference = reference / max_val
        rendered = rendered / max_val  # Normalize both by the same value
    ssim_value, ssim_map = calculate_ssim(reference, rendered, mask)
    return mse_value, ssim_value, ssim_map, mask, reference, rendered



def main():
    image_path1 = 'D:\\dev\\LiverRenderer\\scenes\\Liver-SingleMesh\\mitsuba3\\outputs\\Mitsuba3\\CPU\\liver-singlemesh.png'
    image_path2 = 'D:\\dev\\learned-subsurface-scattering\\pysrc\\outputs\\vae3d\\render\\scenes\\liver\\0487_FinalSharedLs7Mixed3_AbsSharedSimComplexMixed3\\LearnedLiverModel.png'
    mask_path = "D:\\dev\\LiverRenderer\\scenes\\Liver-SingleMesh\\mitsuba3\\LiveMultiMesh-Mask.png"

    rmse, ssim, ssim_map, mask, reference, rendered = compare_images(image_path1, image_path2, mask_path)
    print(f"MSE: {rmse:.4f}")
    print(f"SSIM: {ssim:.4f}")
    visualize_ssim(ssim_map, ssim, mask, 'resultsMasked/LearnedSSIM.png')
    visualize_rmse(reference, rendered, rmse, mask, 'resultsMasked/LearnedRSME.png')


if __name__ == "__main__":
    OPENCV_IO_ENABLE_OPENEXR = True
    main()
