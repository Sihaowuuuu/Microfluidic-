import shutil
from skimage import io, filters, color
import matplotlib.pyplot as plt
import os
import cv2
from os.path import basename
import tkinter as tk
from tkinter import simpledialog
from PIL import Image
import numpy as np

from Count.Count_the_overlay import image_path
from Test_tubes_segmentation import rotate_portrait_to_landscape, select_two_points_crop

# 增加最大图像大小限制
Image.MAX_IMAGE_PIXELS = None  # 或者设置为一个更大的数字，如: 300000000

def select_segment_area(input_folder, save_folder):
    save_folder = os.path.join(save_folder,"semi_segment")
    points = []
    threshold_ch01, threshold_ch02 = get_threshold_value(input_folder)
    # 加载图片
    for semi_cropped_images in os.listdir(input_folder):
        if "semi_cropped_images" not in semi_cropped_images:
            continue

        semi_cropped_images_folder= os.path.join(input_folder,semi_cropped_images)
        for time in os.listdir(semi_cropped_images_folder):
            if "t00" not in time:
                continue

            time_folder = os.path.join(semi_cropped_images_folder,time)
            for image_path in os.listdir(time_folder):

                if "ch00" not in image_path:
                    continue

                image_path = os.path.join(time_folder,image_path)
                img = Image.open(image_path)

                # 获取图片的尺寸
                width, height = img.size

                # 获取左八分之一区域
                left_eighth = img.crop((0, 0, width // 8, height))

                # 展示左八分之一区域
                fig, ax = plt.subplots()
                ax.imshow(left_eighth)
                ax.set_title("Left One-Eighth of Image")
                ax.axis('on')

    # 鼠标点击回调函数
    def on_click(event):
        if len(points) < 2:  # 限制最多选择两个点
            points.append((event.xdata, event.ydata))
            ax.scatter(event.xdata, event.ydata, color='red', s=20)  # 绘制红点标记点击位置
            plt.draw()  # 更新图像
            if len(points) == 2:
                plt.close()

    # 连接鼠标点击事件
    fig.canvas.mpl_connect('button_press_event', on_click)

    plt.show()  # 显示图像并等待用户选择两点

    if len(points) == 2:  # 确保用户选择了两个点
        print(f"Selected points: {points}")

        # 创建一个Tkinter窗口来显示输入框
        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口

        # 使用Tkinter输入框获取Trap Number
        trap_number = simpledialog.askinteger("Input", "Enter Trap Number:", minvalue=1)

        if trap_number:
            print(f"Trap number set to: {trap_number}")

            # 获取两点的坐标并裁剪
            x1, y1 = points[0]
            x2, y2 = points[1]

            # 计算裁剪区域
            left, upper = min(x1, x2), min(y1, y2)
            right, lower = max(x1, x2), max(y1, y2)

            #裁剪有用区域
            for time in os.listdir(semi_cropped_images_folder):
                time_folder = os.path.join(semi_cropped_images_folder, time)
                for image_path in os.listdir(time_folder):
                    # Split the file name by underscore
                    parts = os.path.splitext(image_path)[0].split("_")

                    # Identify the time part (e.g., 'ch00', 'ch01')
                    channel = None
                    for part in parts:
                        if part.startswith("ch") and part[2:].isdigit():  # Check if it matches the "ch" + digits pattern
                            channel = part
                            break

                    if not channel:
                        print(f"Skipping '{image_path}': no valid channel part found.")
                        continue

                    image_path = os.path.join(time_folder, image_path)
                    img = Image.open(image_path)
                    # 从左八分之一区域裁剪中间部分
                    cropped = img.crop((left, upper, right, lower))

                    # 为每次裁剪创建新的文件夹并保存裁剪后的图像
                    save_time_folder = os.path.join(save_folder, time)
                    part_folder = os.path.join(save_time_folder, f"part_1")
                    os.makedirs(part_folder, exist_ok=True)
                    save_path = os.path.join(part_folder, f"segment_area_1_{time}_{channel}.tif")
                    cropped.save(save_path)
                    print(f"Image cropped and saved at {save_path}")
                # 支持的图片扩展名
                image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.gif'}



                for image_path in os.listdir(part_folder):
                    source_path = os.path.join(part_folder,image_path)
                    if os.path.isfile(source_path) and os.path.splitext(image_path)[1].lower() in image_extensions:
                        # 继续后续操作，例如图像分析
                        if "ch00" in image_path and "t00" in image_path:
                            segment_coordinates = scan_low_intensity_on_segment(source_path, trap_number)
                            semi_segmenting(segment_coordinates, source_path, part_folder,0)

                for image_path in os.listdir(part_folder):
                    source_path = os.path.join(part_folder, image_path)
                    if os.path.isfile(source_path) and os.path.splitext(image_path)[1].lower() in image_extensions:
                        # all_segmented_coordinates.append(segment_coordinates)
                        tube_count_next = semi_segmenting(segment_coordinates, source_path, part_folder, 0)

                        semi_threshold_segmenting(segment_coordinates, part_folder, part_folder, 0, threshold_ch01, threshold_ch02)

            crop_width = x2- x1
                # 展示从 x2 开始，向右展示相同长度的区域

            display_right_part(semi_cropped_images_folder, x2, crop_width, width, height, save_folder,2, tube_count_next, threshold_ch01, threshold_ch02)

def display_right_part(input_folder, x2, crop_width, width, height, save_folder, part_number, tube_count, threshold_ch01, threshold_ch02):
    """
    展示从 x2 开始，向右裁剪相同宽度的部分
    """
    points = []
    # 如果 x2 达到或超过图片宽度，终止递归
    if x2 >= width:
        print(f"Stopping recursion: x2={x2}, width={width}")
        return

    start_position = x2
    for time in os.listdir(input_folder):
        if "t00" not in time:
            continue
        time_folder = os.path.join(input_folder, time)
        for image_path in os.listdir(time_folder):
            if "ch00" not in image_path:
                continue

            image_path = os.path.join(time_folder, image_path)
            img = Image.open(image_path)

        # 裁剪右侧区域
        if x2 + crop_width < width:
            right_part = img.crop((x2 - 10, 0, x2 + int(crop_width), height))
        else:
            right_part = img.crop((x2 - 10, 0, width, height))

        # 展示裁剪的右侧区域
        fig, ax = plt.subplots()
        ax.imshow(right_part)
        ax.set_title(f"Part {part_number}: x2={x2} to {x2 + int(crop_width)}")
        ax.axis('on')

    # 鼠标点击回调函数，选择两个点
    def on_click(event):
        if len(points) < 2:  # 限制最多选择两个点
            points.append((event.xdata, event.ydata))
            ax.scatter(event.xdata, event.ydata, color='red', s=20)  # 绘制红点标记点击位置
            plt.draw()

            if len(points) == 2:  # 选择了两个点
                plt.close()  # 关闭当前图像窗口

    fig.canvas.mpl_connect('button_press_event', on_click)

    # 显示图像并等待用户点击选择两个点
    plt.show()

    if len(points) == 2:  # 确保用户选择了两个点
        print(f"Selected points: {points}")

        # 使用Tkinter输入框获取Trap Number
        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口
        trap_number = simpledialog.askinteger("Input", "Enter Trap Number:", minvalue=1)

        if trap_number:
            print(f"Trap number set to: {trap_number}")

            # 获取两点的坐标并裁剪
            x1, y1 = points[0]
            x2, y2 = points[1]

            # 计算裁剪区域
            left, upper = start_position-10 + min(x1, x2), min(y1, y2)
            right, lower = start_position-10 + max(x1, x2), max(y1, y2)

            #裁剪有用区域
            for time in os.listdir(input_folder):
                time_folder = os.path.join(input_folder,time)
                for image_path in os.listdir(time_folder):
                    # Split the file name by underscore
                    parts = os.path.splitext(image_path)[0].split("_")

                    # Identify the time part (e.g., 'ch00', 'ch01')
                    channel = None
                    for part in parts:
                        if part.startswith("ch") and part[2:].isdigit():  # Check if it matches the "t" + digits pattern
                            channel = part
                            break

                    if not channel:
                        print(f"Skipping '{image_path}': no valid time part found.")
                        continue

                    image_path = os.path.join(time_folder,image_path)
                    img = Image.open(image_path)
                    # 新计算的值
                    cropped = img.crop((left, upper, right, lower))

                    # 为每次裁剪创建新的文件夹并保存裁剪后的图像
                    save_time_folder = os.path.join(save_folder, time)
                    part_folder = os.path.join(save_time_folder, f"part_{part_number}")
                    os.makedirs(part_folder, exist_ok=True)
                    save_path = os.path.join(part_folder, f"segment_area_{part_number}_{time}_{channel}.tif")
                    cropped.save(save_path)
                    print(f"Image cropped and saved at {save_path}")

                # 支持的图片扩展名
                image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.gif'}

                for image_path in os.listdir(part_folder):
                    source_path = os.path.join(part_folder, image_path)
                    if os.path.isfile(source_path) and os.path.splitext(image_path)[1].lower() in image_extensions:
                        # 继续后续操作，例如图像分析
                        if "ch00" in image_path and "t00" in image_path:
                            segment_coordinates = scan_low_intensity_on_segment(source_path, trap_number)
                            semi_segmenting(segment_coordinates, source_path, part_folder, tube_count)

                for image_path in os.listdir(part_folder):
                    source_path = os.path.join(part_folder, image_path)
                    if os.path.isfile(source_path) and os.path.splitext(image_path)[1].lower() in image_extensions:
                        tube_count_next = semi_segmenting(segment_coordinates, source_path, part_folder, tube_count)
                        semi_threshold_segmenting(segment_coordinates,part_folder,part_folder,tube_count,threshold_ch01, threshold_ch02)
                # 更新 x2 为当前裁剪区域的末尾
                new_x2 = int(x2) +start_position
                print(f"Next part: new_x2={new_x2}, crop_width={crop_width}")

            if new_x2 < width-10:
                # 继续递归
                display_right_part(input_folder, new_x2, crop_width, width, height, save_folder, part_number + 1, tube_count_next, threshold_ch01, threshold_ch02)


def scan_low_intensity_on_segment(image_path, number):

    segment_coordinates = None
    target_low_intensity_count = number*2
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image in scan_low_intensity_on_segment")
        return
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    initial_y = round(height / 2)
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

    print(final_mean + final_mean / 5)
    step_size = 1
    found = False
    direction = 'down'  # the initial search direction is downward

    while not found:
        if y < 0 or y >= height:
            if direction == 'down':
                # if failed to search downward, then change it to search upward
                print(f"Switching to upward search for ch00")
                y = round(height / 2) - 70
                direction = 'up'
                continue
            else:
                # if search upward also failed, then end searching
                print(f"Unable to find sufficient low-intensity points in ch00")
                segment_coordinates = None
                break

        # Convert row to grayscale and find low-intensity regions
        intensity_values = cv2.cvtColor(image[y:y + 1, :, :], cv2.COLOR_BGR2GRAY)[0]
        segment_coordinates = find_reasonable_low_intensity_pixels(intensity_values,
                                                                   target_low_intensity_count, final_mean+final_mean/5)

        if len(segment_coordinates) == target_low_intensity_count:
            print(f"Found {target_low_intensity_count} low-intensity points at y = {y}")
            found = True
        else:
            # change the searching direction based on the current searching direction
            y += step_size if direction == 'down' else -step_size
    return segment_coordinates

def semi_segmenting(segmented_coordinates, image_path, output_folder, tube):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image.")
        return  # Skip this image if it can't be loaded

    print(f"Processing image: {basename(image_path)}")
    height, width = image.shape[:2]

    # Create a new folder in output folder, named after the image with '_1024' added
    image_name = os.path.splitext(basename(image_path))[0]  # Get the image name without extension
    output_dir = os.path.join(output_folder, f"semi_segmented_{image_name}")  # Add '_1024' suffix to the image name
    os.makedirs(output_dir, exist_ok=True)

    for i in range(0, len(segmented_coordinates) - 1, 2):
        if i + 1 >= len(segmented_coordinates):  # 检查索引越界
            print(f"Invalid index pair: {i}, {i + 1}")
            continue
        start_x = max(0, segmented_coordinates[i] - 2)
        end_x = min(width, segmented_coordinates[i + 1] + 15)

        tube +=1
        # Crop the region of interest (ROI)
        roi = image[:, start_x:end_x]
        print(f"Start_x: {start_x}, End_x: {end_x}, ROI shape: {roi.shape if roi is not None else 'None'}")

        output_filename = f"{tube}.tif"  # Add 1024 suffix to the output filename
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, roi)  # Save the cropped image in color

    print(f"Processing complete. Processed images saved in {output_dir}")
    return tube

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


def gather_all_semi_segment_part(input_folder):
    for semi_segment_folder in os.listdir(input_folder):
        if "semi_segment" not in semi_segment_folder:
            continue

        semi_segment_folder = os.path.join(input_folder, semi_segment_folder)
        output_folder = os.path.join(semi_segment_folder, "all")
        os.makedirs(output_folder, exist_ok=True)

        for time in os.listdir(semi_segment_folder):
            if "t" not in time:
                continue
            time_folder = os.path.join(semi_segment_folder,time)

            for part_folder in os.listdir(time_folder):
                if "part" not in part_folder:
                    continue
                part_folder = os.path. join(time_folder, part_folder)
                for semi_segmented_folder in os.listdir(part_folder):
                    if "semi_segmented" in semi_segmented_folder:
                        semi_segmented_folder_ch = os.path.join(part_folder, semi_segmented_folder)
                        for file in os.listdir(semi_segmented_folder_ch):
                            channel_parts= semi_segmented_folder.split('_')

                            # 通过遍历找到 chXX 这样的字符串
                            channel = None
                            for part in channel_parts:
                                if part.startswith("ch") and part[2:].isdigit():  # 确保 "ch" 后跟的是数字
                                    channel = part
                                    break
                            time_folder_save = os.path.join(output_folder, time)
                            os.makedirs(time_folder_save, exist_ok=True)
                            channel_folder_path = os.path.join(time_folder_save,f"Segmented_{channel}")
                            os.makedirs(channel_folder_path, exist_ok=True)
                            destination_path = os.path.join(channel_folder_path, file)
                            file = os.path.join(semi_segmented_folder_ch, file)

                            shutil.copy2(file, destination_path)
                            print(f"Copied: {file} to {destination_path}")

                    if "Semi_Thresholded_Segmented_segment_area" in semi_segmented_folder:
                        semi_segmented_folder_ch = os.path.join(part_folder, semi_segmented_folder)
                        for file in os.listdir(semi_segmented_folder_ch):
                            channel_parts = semi_segmented_folder.split('_')

                            # 通过遍历找到 chXX 这样的字符串
                            channel = None
                            for part in channel_parts:
                                if part.startswith("ch") and part[2:].isdigit():  # 确保 "ch" 后跟的是数字
                                    channel = part
                                    break
                            time_folder_save = os.path.join(output_folder, time)
                            os.makedirs(time_folder_save, exist_ok=True)
                            channel_folder_path = os.path.join(time_folder_save,f"Thresholded_segmented_{channel}")
                            os.makedirs(channel_folder_path, exist_ok=True)
                            destination_path = os.path.join(channel_folder_path, file)
                            file = os.path.join(semi_segmented_folder_ch, file)

                            shutil.copy2(file, destination_path)
                            print(f"Copied: {file} to {destination_path}")

def get_threshold_value(input_folder):
    max_otsu_threshold_ch01, max_otsu_threshold_ch02 = 0, 0
    for cropped_image in os.listdir(input_folder):
        if "semi_cropped_images" not in cropped_image:
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
                print(max_otsu_threshold_ch01)


            for image in os.listdir(time_folder):
                if "ch02" not in image:
                    continue
                image_path = os.path.join(time_folder,image)
                img = io.imread(image_path)

                gray_image = color.rgb2gray(img) if img.ndim ==3 else img

                otsu_threshold_ch02 = filters.threshold_otsu(gray_image)
                max_otsu_threshold_ch02 = max(otsu_threshold_ch02, max_otsu_threshold_ch02)
                print(max_otsu_threshold_ch02)
    return max_otsu_threshold_ch01, max_otsu_threshold_ch02

def threshold(image_path, threshold_value):
    # Load the image
    image = io.imread(image_path)

    # Convert to grayscale if needed
    gray_image = color.rgb2gray(image) if image.ndim == 3 else image

    binary_image = gray_image > threshold_value

    return binary_image


def semi_threshold_segmenting(overlay_coordinates, input_folder, output_folder, tube, threshold_ch01, threshold_ch02):
    tube_start = tube
    for filename in os.listdir(input_folder):
        if not filename.endswith(('.jpg', '.png', '.tif', '.tiff')):  # Skip non-image files
            continue

        if "ch01" in filename.lower():

            image_path = os.path.join(input_folder, filename)
            image = threshold(image_path, threshold_ch01)

            if image is None:
                print(f"Failed to load image: {image_path}")
                continue  # Skip this image if it can't be loaded

            print(f"Processing image: {filename}")
            height, width = image.shape[:2]

            # Create a new folder in output folder, named after the image with '_1024' added
            image_name = os.path.splitext(filename)[0]  # Get the image name without extension
            output_dir = os.path.join(output_folder, f"Semi_Thresholded_Segmented_{image_name}")  # Add '_1024' suffix to the image name
            os.makedirs(output_dir, exist_ok=True)

            for i in range(0, len(overlay_coordinates) - 1, 2):
                start_x = max(0, overlay_coordinates[i] - 2)
                end_x = min(width - 1, overlay_coordinates[i + 1] + 15)

                # Crop the region of interest (ROI)
                roi = image[:, start_x:end_x]
                tube +=1
                output_filename = f"{tube}.tif"  # Add 1024 suffix to the output filename
                output_path = os.path.join(output_dir, output_filename)
                cv2.imwrite(output_path, (roi * 255).astype('uint8'))  # Save the cropped image in color
                print(f"Processing complete. Processed images saved in {output_dir}")

        if "ch02" in filename.lower():
            tube = tube_start
            image_path = os.path.join(input_folder, filename)
            image = threshold(image_path,threshold_ch02)

            if image is None:
                print(f"Failed to load image: {image_path}")
                continue  # Skip this image if it can't be loaded

            print(f"Processing image: {filename}")
            height, width = image.shape[:2]

            # Create a new folder in output folder, named after the image with '_1024' added
            image_name = os.path.splitext(filename)[0]  # Get the image name without extension
            output_dir = os.path.join(output_folder,
                                      f"Semi_Thresholded_Segmented_{image_name}")  # Add '_1024' suffix to the image name
            os.makedirs(output_dir, exist_ok=True)

            for i in range(0, len(overlay_coordinates) - 1, 2):
                start_x = max(0, overlay_coordinates[i] - 2)
                end_x = min(width - 1, overlay_coordinates[i + 1] + 15)

                # Crop the region of interest (ROI)
                roi = image[:, start_x:end_x]
                tube +=1
                output_filename = f"{tube}.tif"  # Add 1024 suffix to the output filename
                output_path = os.path.join(output_dir, output_filename)
                cv2.imwrite(output_path, (roi * 255).astype('uint8'))  # Save the cropped image in color

                print(f"Processing complete. Processed images saved in {output_dir}")

def crop_two_sides(input_folder, output_folder, points):
    # Calculate the crop boundaries
    pt1, pt2 = points

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    for straightened_folder in sorted(os.listdir(input_folder)):
        if "straightened image" not in straightened_folder:
            continue

        straightened_folder = os.path.join(input_folder, straightened_folder)

        for time in os.listdir(straightened_folder):
            time_folder = os.path.join(straightened_folder,time)

            # Process all images in the folder
            for filename in os.listdir(time_folder):
                image_path = os.path.join(time_folder, filename)
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)

                # Rotate the image to landscape
                landscape_image = rotate_portrait_to_landscape(image)

                # Convert to RGB
                image_rgb = cv2.cvtColor(landscape_image, cv2.COLOR_BGR2RGB)

                # Crop only the left and right boundaries
                crop_left = min(pt1[0], pt2[0])  # 左边界
                crop_right = max(pt1[0], pt2[0])  # 右边界

                # Convert cropped image to PIL object
                image_pil = Image.fromarray(image_rgb)
                cropped_image = image_pil.crop((crop_left, 0, crop_right, image_pil.height))

                # Create output subfolder if it doesn't exist
                subfolder_path = os.path.join(output_folder, f"semi_cropped_images")
                os.makedirs(subfolder_path, exist_ok=True)

                subfolder_path = os.path.join(subfolder_path, time)
                os.makedirs(subfolder_path, exist_ok=True)

                # Save the processed image in the subfolder
                output_path = os.path.join(subfolder_path, f"semi_cropped_{filename}")
                cropped_image.save(output_path)
                print(f"Cropped image saved at: {output_path}")

input_folder = r"D:\thesis\processed 7th\R3"
output_folder = r"D:\thesis\processed 7th\R2\Scale"
#crop_two_sides(*select_two_points_crop(input_folder,output_folder))
#select_segment_area(input_folder, output_folder)
gather_all_semi_segment_part(input_folder)