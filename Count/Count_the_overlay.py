from skimage import io, color, filters, morphology, segmentation
from skimage.feature import peak_local_max
from skimage.measure import regionprops_table
from scipy.ndimage import distance_transform_cdt
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi


def count_overlay(image_path, red_intensity_threshold=180, green_intensity_threshold=180):
    """
    Count fluorescent cells with intensity thresholds for both red and green channels.

    Args:
        image_path (str): Path to the input image.
        red_intensity_threshold (int): Minimum intensity for the red channel. Default is 180.
        green_intensity_threshold (int): Minimum intensity for the green channel. Default is 180.

    Returns:
        int: Number of fluorescent cells detected.
    """
    # Load the image
    image = io.imread(image_path)

    # Ensure the image is in RGB
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("The input image must be an RGB image.")

    # Extract the red and green channels
    red_channel = image[:, :, 0]
    green_channel = image[:, :, 1]

    # Apply intensity thresholds
    filtered_red_channel = red_channel.copy()
    filtered_red_channel[red_channel < red_intensity_threshold] = 0

    filtered_green_channel = green_channel.copy()
    filtered_green_channel[green_channel < green_intensity_threshold] = 0

    """
      # Visualize the original image and filtered channels
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(red_channel, cmap="Reds")
    axes[1].set_title("Red Channel")
    axes[1].axis("off")

    axes[2].imshow(filtered_red_channel, cmap="Reds")
    axes[2].set_title(f"Filtered Red (Threshold: {red_intensity_threshold})")
    axes[2].axis("off")

    axes[3].imshow(green_channel, cmap="Greens")
    axes[3].set_title("Green Channel")
    axes[3].axis("off")

    axes[4].imshow(filtered_green_channel, cmap="Greens")
    axes[4].set_title(f"Filtered Green (Threshold: {green_intensity_threshold})")
    axes[4].axis("off")

    plt.tight_layout()
    plt.show()
    """

    # Threshold the filtered channels
    red_mask = filtered_red_channel > filters.threshold_otsu(filtered_red_channel)
    green_mask = filtered_green_channel > filters.threshold_otsu(filtered_green_channel)

    # Combine red and green masks
    combined_mask = red_mask | green_mask

    # Morphological manipulation to remove noise (small spots)
    cleaned_image = morphology.remove_small_objects(combined_mask, min_size=100)  # Set the minimum size of particles
    cleaned_image = morphology.remove_small_holes(cleaned_image, area_threshold=1)  # Fill the small holes

    # Calculate the distance transform
    distance_map = distance_transform_cdt(cleaned_image, metric='taxicab')

    # Find local maxima and generate markers that match the shape of the image
    local_max = peak_local_max(
        distance_map,
        min_distance=7,
        footprint=np.ones((3, 3)),
        labels=cleaned_image
    )

    # Convert local maxima to boolean masks
    local_max_mask = np.zeros_like(distance_map, dtype=bool)
    local_max_mask[tuple(local_max.T)] = True

    # Using the Markup Generator
    markers = ndi.label(local_max_mask)[0]

    # Segmenting adherent particles with Watershed
    segmented_image = segmentation.watershed(-distance_map, markers, mask=cleaned_image)

    # Extracting particle property data
    properties = regionprops_table(
        segmented_image, properties=('label', 'area', 'perimeter', 'centroid')
    )

    # Return data and segmentation results
    return len(np.unique(segmented_image)) - 1  # Subtract 1 for background

# Example usage
image_path = r"D:\thesis\processed 6th new\R1\t00\Segmented_straightened_20241208_U87_NK92_6th_try_1024_TileScan 2_R 1_Merged__t00_t00_ch01_SV_1024\straightened_20241208_U87_NK92_6th_try_1024_TileScan 2_R 1_Merged__t00_t00_ch01_SV_1024_7.jpg"
#cell_count = count_overlay(image_path, red_intensity_threshold=180, green_intensity_threshold=180)
#print(cell_count)




