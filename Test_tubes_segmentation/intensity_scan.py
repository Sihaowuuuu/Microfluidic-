import os
import cv2
from skimage import io, filters, color
import numpy as np

# Filter to find low-intensity pixel positions with set distances
def find_reasonable_low_intensity_pixels(intensity_values, target_count, final_mean):
    low_intensity_indices = []
    distance_pattern = [16, 16]
    pattern_index = 0

    for i in range(len(intensity_values)):
        if intensity_values[i] < final_mean:  # Intensity threshold
            if not low_intensity_indices or (100 >i - low_intensity_indices[-1] > distance_pattern[pattern_index]):
                low_intensity_indices.append(i)
                pattern_index = (pattern_index + 1) % len(distance_pattern)
                if len(low_intensity_indices) == target_count:
                    break

    return low_intensity_indices

# Function to detect the edge of test tubes from images based on overlay
def scan_low_intensity_on_overlay(input_folder, output_folder, number):
    os.makedirs(output_folder, exist_ok=True)

    overlay_coordinates = None
    target_low_intensity_count = number*2
    # Use os.walk() to check through the input folder and its subfolders
    for cropped_folder in sorted(os.listdir(input_folder)):
        if cropped_folder != "cropped image":
            continue
        cropped_folder = os.path.join(input_folder, cropped_folder)
        for root, dirs, files in os.walk(cropped_folder):  # root is the current folder, dirs are subfolders, files are files
            for filename in sorted(files):
                if not filename.endswith(('.tif', '.tiff')):  # Skip non-image files
                    continue

                image_path = os.path.join(root, filename)
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                if image is None:
                    print(f"Failed to load image: {image_path}")
                    continue

                # Output folder for the current image, create subfolders for each file's path
                relative_path = os.path.relpath(root, input_folder)  # Get relative path from input_folder
                output_dir = os.path.join(output_folder, relative_path)
                os.makedirs(output_dir, exist_ok=True)

                # If filename contains "ch00", detect low-intensity regions
                if 'ch00' in filename and "t00" in filename:
                    print(f"Processing overlay image: {filename}")
                    height, width = image.shape[:2]
                    initial_y = round(height / 2) + 20
                    y = initial_y
                    intensity_values = cv2.cvtColor(image[y:y + 1, :, :], cv2.COLOR_BGR2GRAY)[0]
                    # 第一次计算平均值
                    mean1 = np.mean(intensity_values)

                    # 第一次筛选：低于 mean1 的点
                    low_intensity_values1 = intensity_values[intensity_values < mean1]

                    # 确保第一次筛选结果不为空
                    if low_intensity_values1.size > 0:
                        # 第二次计算平均值
                        mean2 = np.mean(low_intensity_values1)

                        # 第二次筛选：低于 mean2 的点
                        low_intensity_values2 = low_intensity_values1[low_intensity_values1 < mean2]

                        # 确保第二次筛选结果不为空
                        final_mean = np.mean(low_intensity_values2) if low_intensity_values2.size > 0 else mean2
                    else:
                        final_mean = mean1  # 如果第一次筛选为空，直接返回 mean1

                    print(final_mean+final_mean/5)
                    step_size = 1
                    found = False
                    direction = 'down'  # the initial search direction is downward

                    while not found:
                        if y < 0 or y >= height:
                            if direction == 'down':
                                # if failed to search downward, then change it to search upward
                                print(f"Switching to upward search for channel 0: {filename}")
                                y = round(height / 2) - 70
                                direction = 'up'
                                continue
                            else:
                                # if search upward also failed, then end searching
                                print(f"Unable to find sufficient low-intensity points in channel 0: {image_path}")
                                overlay_coordinates = None
                                break

                        # Convert row to grayscale and find low-intensity regions
                        intensity_values = cv2.cvtColor(image[y:y + 1, :, :], cv2.COLOR_BGR2GRAY)[0]
                        overlay_coordinates = find_reasonable_low_intensity_pixels(intensity_values,
                                                                                   target_low_intensity_count, final_mean+final_mean/5)

                        if len(overlay_coordinates) == target_low_intensity_count:
                            print(f"Found {target_low_intensity_count} low-intensity points at y = {y}")
                            found = True
                        else:
                            # change the searching direction based on the current searching direction
                            y += step_size if direction == 'down' else -step_size
        return overlay_coordinates


def segmenting(overlay_coordinates, input_folder, output_folder):
    # If overlay coordinates are detected, apply cropping to all images
    for cropped_folder in sorted(os.listdir(input_folder)):
        if "cropped image" not in cropped_folder:
            continue
        cropped_folder = os.path.join(input_folder, cropped_folder)

        for root, dirs, files in os.walk(cropped_folder):  # root is the current folder, dirs are subfolders, files are files
            for filename in sorted(files):
                if not filename.endswith(('.jpg', '.png', '.tif', '.tiff')):  # Skip non-image files
                    continue

                image_path = os.path.join(root, filename)
                image = cv2.imread(image_path)

                if image is None:
                    print(f"Failed to load image: {image_path}")
                    continue  # Skip this image if it can't be loaded

                print(f"Processing image: {filename}")
                height, width = image.shape[:2]

                # Create a new folder in output folder, named after the image with '_1024' added
                image_name = os.path.splitext(filename)[0]  # Get the image name without extension
                time_name = os.path.basename(root)
                save_folder = os.path.join(output_folder, time_name)
                output_dir = os.path.join(save_folder, f"Segmented_{image_name}_1024")  # Add '_1024' suffix to the image name
                os.makedirs(output_dir, exist_ok=True)

                if not os.access(output_dir, os.W_OK):
                    print(f"错误: 无写入权限 {output_dir}")
                    continue

                for i in range(0, len(overlay_coordinates) - 1, 2):
                    if i + 1 >= len(overlay_coordinates):  # 检查索引越界
                        print(f"Invalid index pair: {i}, {i + 1}")
                        continue
                    start_x = max(0, overlay_coordinates[i] - 2)
                    end_x = min(width, overlay_coordinates[i + 1] + 15)

                    # Crop the region of interest (ROI)
                    roi = image[:, start_x:end_x]
                    print(f"Start_x: {start_x}, End_x: {end_x}, ROI shape: {roi.shape if roi is not None else 'None'}")
                    output_filename = f"{i // 2}.tif"  # Add 1024 suffix to the output filename
                    output_path = os.path.join(output_dir, output_filename)
                    try:
                        success = cv2.imwrite(output_path, roi)
                        if not success:
                            # 尝试保存为PNG格式
                            png_path = os.path.splitext(output_path)[0] + ".png"
                            cv2.imwrite(png_path, roi)
                            print(f"警告: TIFF保存失败，已转为PNG格式: {png_path}")
                        else:
                            print(f"保存成功: {output_path}")
                    except Exception as e:
                        print(f"异常: {str(e)}")

                print(f"Processing complete. Processed images saved in {output_dir}")


def get_threshold_value(input_folder):
    max_otsu_threshold_ch01, max_otsu_threshold_ch02 = 0, 0
    for cropped_image in os.listdir(input_folder):
        if "cropped image" not in cropped_image:
            continue
        cropped_image_folder = os.path.join(input_folder,cropped_image)
        for time in os.listdir(cropped_image_folder):
            time_folder = os.path.join(cropped_image_folder,time)
            for image in os.listdir(time_folder):
                if "ch01" not in image:
                    continue
                image_path = os.path.join(time_folder,image)
                img = io.imread(image_path)

                gray_image = color.rgb2gray(img) if img.ndim ==3 else img

                otsu_threshold_ch01 = filters.threshold_otsu(gray_image)
                max_otsu_threshold_ch01 = max(otsu_threshold_ch01, max_otsu_threshold_ch01)
                print(f"max_otsu_threshold_ch01:{max_otsu_threshold_ch01}")


            for image in os.listdir(time_folder):
                if "ch02" not in image:
                    continue
                image_path = os.path.join(time_folder,image)
                img = io.imread(image_path)

                gray_image = color.rgb2gray(img) if img.ndim ==3 else img

                otsu_threshold_ch02 = filters.threshold_otsu(gray_image)
                max_otsu_threshold_ch02 = max(otsu_threshold_ch02, max_otsu_threshold_ch02)
                print(f"max_otsu_threshold_ch02:{max_otsu_threshold_ch02}")
    return max_otsu_threshold_ch01, max_otsu_threshold_ch02

def threshold(image_path, threshold_value):
    # Load the image
    image = io.imread(image_path)

    # Convert to grayscale
    gray_image = color.rgb2gray(image) if image.ndim == 3 else image

    binary_image = gray_image > threshold_value  # Use the manual threshold

    return binary_image


def threshold_segmenting(overlay_coordinates, input_folder, output_folder):
    # If overlay coordinates are detected, apply cropping to all images
    otsu_threshold_ch01, otsu_threshold_ch02 = get_threshold_value(input_folder)
    threshold_input_folder = os.path.join(input_folder,f"cropped image")
    for root, dirs, files in os.walk(threshold_input_folder):  # root is the current folder, dirs are subfolders, files are files
        for filename in sorted(files):
            if not filename.endswith(('.tif', '.tiff')):  # Skip non-image files
                continue

            if "ch01" in filename.lower():

                image_path = os.path.join(root, filename)
                image = threshold(image_path, otsu_threshold_ch01)

                if image is None:
                    print(f"Failed to load image: {image_path}")
                    continue  # Skip this image if it can't be loaded

                print(f"Processing image: {filename}")
                height, width = image.shape[:2]

                # Create a new folder in output folder, named after the image with '_1024' added
                image_name = os.path.splitext(filename)[0]  # Get the image name without extension
                time_name = os.path.basename(root)
                save_folder = os.path.join(output_folder,time_name)
                output_dir = os.path.join(save_folder, f"Thresholded_Segmented_{image_name}_1024")  # Add '_1024' suffix to the image name
                os.makedirs(output_dir, exist_ok=True)

                for i in range(0, len(overlay_coordinates) - 1, 2):
                    start_x = max(0, overlay_coordinates[i] - 2)
                    end_x = min(width - 1, overlay_coordinates[i + 1] + 15)

                    # Crop the region of interest (ROI)
                    roi = image[:, start_x:end_x]
                    output_filename = f"{i // 2}.jpg"  # Add 1024 suffix to the output filename
                    output_path = os.path.join(output_dir, output_filename)
                    cv2.imwrite(output_path, (roi * 255).astype(np.uint8))  # Save the cropped image in color
                    print(f"Processing complete. Processed images saved in {output_dir}")

            if "ch02" in filename.lower():

                image_path = os.path.join(root, filename)
                image = threshold(image_path, otsu_threshold_ch02)

                if image is None:
                    print(f"Failed to load image: {image_path}")
                    continue  # Skip this image if it can't be loaded

                print(f"Processing image: {filename}")
                height, width = image.shape[:2]

                # Create a new folder in output folder, named after the image with '_1024' added
                image_name = os.path.splitext(filename)[0]  # Get the image name without extension
                time_name = os.path.basename(root)
                save_folder = os.path.join(output_folder,time_name)
                output_dir = os.path.join(save_folder,
                                          f"Thresholded_Segmented_{image_name}_1024")  # Add '_1024' suffix to the image name
                os.makedirs(output_dir, exist_ok=True)

                for i in range(0, len(overlay_coordinates) - 1, 2):
                    start_x = max(0, overlay_coordinates[i] - 2)
                    end_x = min(width - 1, overlay_coordinates[i + 1] + 15)

                    # Crop the region of interest (ROI)
                    roi = image[:, start_x:end_x]
                    output_filename = f"{i // 2}.jpg"  # Add 1024 suffix to the output filename
                    output_path = os.path.join(output_dir, output_filename)
                    cv2.imwrite(output_path, (roi * 255).astype(np.uint8))  # Save the cropped image in color

                    print(f"Processing complete. Processed images saved in {output_dir}")

# Example usage
input_folder = r"D:\thesis\processed 7th\R2\Scale"  # Replace with your input folder
output_folder = r"D:\thesis\processed 7th\R2\Scale"  # Replace with your output folder

#threshold_segmenting(scan_low_intensity_on_overlay(input_folder, output_folder,1024),input_folder, output_folder)
#segmenting(scan_low_intensity_on_overlay(input_folder, output_folder,1024),input_folder, output_folder)
