import numpy as np
import pandas as pd
from skimage import io, morphology, segmentation
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from scipy.ndimage import distance_transform_cdt
import os
from skimage.measure import regionprops_table

#mage_path = r"D:\thesis\processed 6th new\R1\t00\Thresholded_Segmented_cropped_straightened_20241208_U87_NK92_6th_try_1024_TileScan 2_R 1_Merged__t00_t00_ch01_SV_1024\cropped_straightened_20241208_U87_NK92_6th_try_1024_TileScan 2_R 1_Merged__t00_t00_ch01_SV_1024_75.jpg"
def count_channel(image_path):

    # load the image
    image = io.imread(image_path)

    # Morphological manipulation to remove noise (small spots)
    cleaned_image = morphology.remove_small_objects(image, min_size=100)  # set the minimum size of particles
    cleaned_image = morphology.remove_small_holes(cleaned_image, area_threshold=10)  # fill the small holes

    # calculate the distance transform
    distance_map = distance_transform_cdt(cleaned_image, metric='taxicab')

    # Find local maxima and generate markers that match the shape of the image

    # 在平滑后的距离图上查找局部最大值
    local_max = peak_local_max(
        distance_map,
        min_distance=20,
        footprint=np.ones((3, 3)),
        labels=cleaned_image
    )

    # Convert local maxima to boolean masks
    local_max_mask = np.zeros_like(distance_map, dtype=bool)
    local_max_mask[tuple(local_max.T)] = True

    # Using the Markup Generator
    markers = ndi.label(local_max_mask)[0]

    # Segmenting Adherent Particles with Watershed
    segmented_image = segmentation.watershed(-distance_map, markers, mask=cleaned_image)

    # Extracting particle property data
    properties = regionprops_table(
        segmented_image, properties=('label', 'area', 'perimeter', 'centroid')
    )
    # Return data and segmentation results
    """
    # Plot the results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(distance_map, cmap='viridis')
    axes[0].set_title('Distance Transform')
    axes[0].axis('off')

    # Show local maxima on the original image
    axes[1].imshow(image, cmap='gray')
    axes[1].scatter(local_max[:, 1], local_max[:, 0], color='red', s=5, label='Local Maxima')
    axes[1].set_title('Local Maxima on Original Image')
    axes[1].axis('off')
    axes[1].legend()

    axes[2].imshow(segmented_image, cmap='nipy_spectral')
    axes[2].set_title('Watershed Segmentation')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()
    """
    print(len(np.unique(segmented_image))-1)
    return len(np.unique(segmented_image))-1

# Batch process function
def batch_process_images(input_folder, excel_path):
    results = []
    total_particle_count=0
    intensity_analysis_samples=[]
    # Iterate through all the image files in the folder
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.png', '.tif', '.bmp')):
            image_path = os.path.join(input_folder, filename)
            print(f"Processing {image_path}")
            # Obtaining statistics and segmentation results
            particle_count = count_channel(image_path)

            #Note down the trap with particles for further intensity changing detection
            if particle_count != 0:
                intensity_analysis_samples.append(image_path)

            total_particle_count += particle_count
            results.append({'image_name':filename,'particle_count':particle_count})


    # convert to DataFrame
    final_df = pd.DataFrame(results)

    # Add the total number of particles to the last row of the DataFrame
    final_df = pd.concat([final_df, pd.DataFrame([{'image_name': 'Total', 'particle_count': total_particle_count}])])

    # save to Excel
    excel_path_output = os.path.join(excel_path, "output.xlsx")
    final_df.to_excel(excel_path_output, index=False, engine='openpyxl')
    print(f"Data saved to Excel: {excel_path_output}")

    return intensity_analysis_samples

#input_folder = r"D:\thesis\coding\data\output_folder\straightened_20240213 red green trap scaled channel_P5 combined TileScan 1_Merged_Resize001_ch01_SV_1024"
#excel_path = r"D:\thesis\coding\data\count samples"

#batch_process_images(input_folder, excel_path)

#print(count_channel(image_path))
#count_channel(mage_path)