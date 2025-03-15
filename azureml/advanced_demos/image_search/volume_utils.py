import nibabel as nib


def load_nifti_file(file_path):
    """
    Loads a NIfTI file and returns the volumetric data as a NumPy array.

    Parameters:
    - file_path (str): Path to the NIfTI file (.nii or .nii.gz).

    Returns:
    - numpy.ndarray: 3D array representing the NIfTI file's data.
    """
    nifti_image = nib.load(file_path)
    nifti_data = nifti_image.get_fdata()
    return nifti_data


def normalize_volume(volume):
    """
    Normalizes a 3D volume to the range [0, 255].

    Parameters:
    - volume (numpy.ndarray): 3D array representing the volumetric data.

    Returns:
    - numpy.ndarray: Normalized volume with values between 0 and 255.
    """
    volume[volume < -1000] = -1000
    volume[volume > 1000] = 1000
    normalized_volume = (volume + 1000) / 2000
    normalized_volume = (
        (normalized_volume - normalized_volume.min())
        * 255.0
        / (normalized_volume.max() - normalized_volume.min())
    )
    return normalized_volume


def convert_volume_to_slices(volume, output_dir, filename_prefix):
    """
    Converts a 3D volume into 2D slice images and saves them as PNG files.

    Parameters:
    - volume (numpy.ndarray): 3D array representing the volumetric data.
    - output_dir (str): Directory where slice images will be saved.
    - filename_prefix (str): Prefix for each saved image file.

    Each slice is rotated 90° clockwise and flipped horizontally before saving.
    """
    for i in range(volume.shape[2]):
        slice_img = volume[:, :, i]
        slice_img = slice_img.astype(np.uint8)
        slice_img = (
            Image.fromarray(slice_img)
            .transpose(Image.ROTATE_90)
            .transpose(Image.FLIP_LEFT_RIGHT)
        )
        slice_img.save(os.path.join(output_dir, f"{filename_prefix}_slice{i}.png"))
