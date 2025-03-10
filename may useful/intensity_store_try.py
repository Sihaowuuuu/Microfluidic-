import os

import cv2


# Function to find low-intensity pixel positions with alternating distances
def find_alternating_low_intensity_pixels(intensity_values, target_count):
    low_intensity_indices = []
    distance_pattern = [15, 10]
    pattern_index = 0

    for i in range(len(intensity_values)):
        if intensity_values[i] < 60:  # Intensity threshold for grayscale conversion
            if not low_intensity_indices or (
                i - low_intensity_indices[-1] > distance_pattern[pattern_index]
            ):
                low_intensity_indices.append(i)
                pattern_index = (pattern_index + 1) % len(distance_pattern)
                if len(low_intensity_indices) == target_count:
                    break

    return low_intensity_indices


# Main function to process images based on overlay and apply cropping
def scan_low_intensity_on_overlay(
    input_folder, output_folder, target_low_intensity_count=2048
):
    os.makedirs(output_folder, exist_ok=True)

    overlay_coordinates = None  # Initialize coordinates

    for filename in sorted(
        os.listdir(input_folder)
    ):  # Sort filenames for consistent processing
        if not filename.endswith(
            (".jpg", ".png", ".tif", ".tiff")
        ):  # Skip non-image files
            continue

        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue

        # Output folder for the current image
        output_dir = os.path.join(output_folder, os.path.splitext(filename)[0])
        os.makedirs(output_dir, exist_ok=True)

        # If filename contains "overlay", detect low-intensity regions
        if "overlay" in filename:
            print(f"Processing overlay image: {filename}")
            height, width = image.shape[:2]
            initial_y = round(height / 2) + 90
            y = initial_y
            step_size = 1
            found = False

            while not found:
                if y < 0 or y >= height:
                    print(
                        f"Unable to find sufficient low-intensity points in overlay: {image_path}"
                    )
                    overlay_coordinates = None
                    break

                # Convert row to grayscale and find low-intensity regions
                intensity_values = cv2.cvtColor(
                    image[y : y + 1, :, :], cv2.COLOR_BGR2GRAY
                )[0]
                overlay_coordinates = find_alternating_low_intensity_pixels(
                    intensity_values, target_low_intensity_count
                )

                if len(overlay_coordinates) == target_low_intensity_count:
                    print(
                        f"Found {target_low_intensity_count} low-intensity points at y = {y}"
                    )
                    found = True
                else:
                    y += (
                        step_size
                        if len(overlay_coordinates) < target_low_intensity_count
                        else -step_size
                    )
    return overlay_coordinates


def segmenting(overlay_coordinates, input_folder, output_folder):
    # If overlay coordinates are detected, apply cropping to all images
    for filename in sorted(os.listdir(input_folder)):
        if not filename.endswith(
            (".jpg", ".png", ".tif", ".tiff")
        ):  # Skip non-image files
            continue

        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        if image is None:
            print(f"Failed to load image: {image_path}")
            continue  # Skip this image if it can't be loaded

        print(f"Processing image: {filename}")
        height, width = image.shape[:2]
        # Output folder for the current image
        output_dir = os.path.join(output_folder, os.path.splitext(filename)[0])
        os.makedirs(output_dir, exist_ok=True)

        for i in range(0, len(overlay_coordinates) - 1, 2):
            start_x = max(0, overlay_coordinates[i] - 2)
            end_x = min(width - 1, overlay_coordinates[i + 1] + 5)

            # Crop the region of interest (ROI)
            roi = image[:, start_x:end_x]
            output_filename = (
                f"{os.path.splitext(filename)[0]}_1024_{i // 2}.jpg"  # Add 1024 suffix
            )
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, roi)  # Save the cropped image in color

        print(f"Processing complete. Processed images saved in {output_folder}")


# Example usage
input_folder = r"D:\thesis\coding\data\output_folder"  # Replace with your input folder
output_folder = (
    r"D:\thesis\coding\data\output_folder"  # Replace with your output folder
)

segmenting(
    scan_low_intensity_on_overlay(input_folder, output_folder),
    input_folder,
    output_folder,
)
