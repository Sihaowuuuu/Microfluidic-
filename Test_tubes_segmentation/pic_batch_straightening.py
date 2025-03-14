import cv2
import numpy as np
import math
from PIL import Image
import os
import matplotlib.pyplot as plt


def rotate_portrait_to_landscape(image):
    """
    Rotates a portrait image (height > width) to landscape (width > height).

    Parameters:
        image (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Rotated image if portrait; original image if already landscape.
    """
    height, width = image.shape[:2]
    if height > width:  # Check if the image is portrait
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)  # Rotate 90 degrees clockwise
    return image  # Return the image unchanged if it's already landscape


def auto_pic_point_select(input_folder, output_folder):
    # Load the template image
    template_path = r"D:\thesis\coding\Test_tubes_segmentation\sand.tif"
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    h, w = template.shape[:2]

    # Define the threshold for template matching
    threshold = 0.4

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # List to accumulate processed images for the stack
    matched_centers = []

    # Save the center coordinates
    saved_centers = []

    # Loop through each image in the input folder
    for time_folder in sorted(os.listdir(input_folder)):
        if "t00" not in time_folder:
            continue

        time_folder = os.path.join(input_folder, time_folder)
        for filename in os.listdir(time_folder):
            print(f"Processing file: {filename}")
            # Only proceed if "overlay" is in the filename and it ends with .tif or .tiff
            if "ch00" in filename.lower() and (
                filename.endswith(".tif") or filename.endswith(".tiff") or filename.endswith(".jpg")
            ):

                image_path = os.path.join(time_folder, filename)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                # Rotate the image to landscape
                landscape_image = rotate_portrait_to_landscape(image)

                # For the overlay image, perform template matching and calculate parameters
                res = cv2.matchTemplate(landscape_image, template, cv2.TM_CCOEFF_NORMED)
                locations = np.where(res >= threshold)

                for pt in zip(*locations[::-1]):
                    x, y = pt
                    center_x = x + w // 2
                    center_y = y + h // 2
                    matched_centers.append((center_x, center_y))

                if len(matched_centers) < 2:
                    raise RuntimeError(
                        "Not enough points for rotation in the first image."
                    )

                # Check all matched center coordinates
                for i, center in enumerate(matched_centers):
                    should_save = True
                    for saved_center in saved_centers:
                        # Calculate the Euclidean distance
                        distance = math.sqrt(
                            (center[0] - saved_center[0]) ** 2
                            + (center[1] - saved_center[1]) ** 2
                        )
                        if distance < 100:
                            should_save = False
                            break

                    # If the distance is far enough, save the image and record the center coordinates
                    if should_save:
                        saved_centers.append(
                            center
                        )  # Add the current center coordinates to the list of saved coordinates.

        # Check if saved_centers contains exact two centers
        if len(saved_centers) < 2:
            raise RuntimeError("Not enough matching centers found for rotation.")

        if len(saved_centers) > 2:
            raise RuntimeError("Too many matching centers found for rotation.")
        return input_folder, output_folder, saved_centers


# Function for batch rotation and cropping of images
def auto_batch_straighten_and_crop(input_folder, output_folder, points):
    pt1, pt2 = points
    dx, dy = pt2[0] - pt1[0], pt2[1] - pt1[1]
    angle = math.degrees(math.atan2(dy, dx))
    x_min, x_max = min(pt1[0], pt2[0]), max(pt1[0], pt2[0])

    # Crop and rotation parameters
    crop_top = pt2[1] - 265
    crop_bottom = pt2[1] + 265
    crop_left = 60
    crop_right = None

    os.makedirs(output_folder, exist_ok=True)

    # Process all images in the folder
    for time_folder in sorted(os.listdir(input_folder)):

        time_folder = os.path.join(input_folder, time_folder)
        for filename in os.listdir(time_folder):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Crop and rotate the image based on selected coordinates
            cropped_image = image_rgb[:, x_min:x_max]
            cropped_image_pil = Image.fromarray(cropped_image)
            rotated_image = cropped_image_pil.rotate(angle, expand=True)

            # Determine the right boundary for cropping
            if crop_right is None:
                crop_right = rotated_image.size[0] - 60

            cropped_rotated_image = rotated_image.crop(
                (crop_left, crop_top, crop_right, crop_bottom)
            )

            # Extract a part of the filename to create subfolder (example: using first part of filename)
            subfolder_name = subfolder_name = os.path.join(
                "straightened image", filename.split("_")[10]
            )  # Modify this as needed to extract a specific part
            subfolder_path = os.path.join(output_folder, subfolder_name)

            # Create subfolder if it doesn't exist
            os.makedirs(subfolder_path, exist_ok=True)

            # Save the processed image in the subfolder
            output_path = os.path.join(subfolder_path, f"straightened_{filename}")
            cropped_rotated_image.save(output_path)
            print(f"Straightened image saved at: {output_path}")


# Function for manual batch rotation
def manual_batch_straighten(input_folder, output_folder, points):
    # Calculate the angle
    pt1, pt2 = points
    dx, dy = pt2[0] - pt1[0], pt2[1] - pt1[1]
    angle = math.degrees(math.atan2(dy, dx))

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    for time in sorted(os.listdir(input_folder)):
        time_folder = os.path.join(input_folder, time)
        # Process all images in the folder
        for filename in os.listdir(time_folder):
            image_path = os.path.join(time_folder, filename)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)

            # Rotate the image to landscape
            landscape_image = rotate_portrait_to_landscape(image)

            # Convert to RGB
            image_rgb = cv2.cvtColor(landscape_image, cv2.COLOR_BGR2RGB)

            # Crop and rotation parameters
            crop_top = pt2[1] - 265
            crop_bottom = pt2[1] + 265
            crop_left = pt1[0] - 50
            crop_right = pt2[0] + 50

            # Convert cropped image to PIL object and rotate
            image_pil = Image.fromarray(image_rgb)
            rotated_image = image_pil.rotate(angle, expand=True)

            # Extract a part of the filename to create subfolder (example: using first character or some part)
            subfolder_name = os.path.join(
                "straightened image", time
            )  # Modify as needed to extract a specific part
            subfolder_path = os.path.join(output_folder, subfolder_name)

            # Create subfolder if it doesn't exist
            os.makedirs(subfolder_path, exist_ok=True)

            # Save the processed image in the subfolder
            output_path = os.path.join(subfolder_path, f"straightened_{filename}")
            rotated_image.save(output_path)
            print(f"Straightened image saved at: {output_path}")


def select_two_points(input_folder, output_folder):
    """
    allow the user to select a point from left side and another point from right side.

    Parameters:
        input_folder (str): the path of the folder of input image
        output_folder (str): the path of the folder of output image

    Returns:
        tuple: (input_folder, output_folder, points)，the form of the points [(x1, y1), (x2, y2)]。
    """
    image_path = None
    points = []

    # find the overlay image
    for time_folder in sorted(os.listdir(input_folder)):
        if "t00" not in time_folder:
            continue
        time_folder = os.path.join(input_folder, time_folder)
        for filename in os.listdir(time_folder):
            if "ch00" in filename.lower() and filename.lower().endswith(
                (".tif", ".tiff", ".jpg")
            ):
                image_path = os.path.join(time_folder, filename)
                break

        if image_path is None:
            print("Error: No valid overlay image found.")
            return None

        # load the image
        image = rotate_portrait_to_landscape(cv2.imread(image_path, cv2.IMREAD_COLOR))
        if image is None:
            print("Error: Could not open image.")
            return None

        height, width = image.shape[:2]

        # Extract 1000 pixels wide from the leftmost and rightmost parts of the image
        left_half = image[:, :1000]
        right_half = image[:, width - 1000 :]

        def select_point(img, title):
            """Displays the image and allows the user to select a point."""
            fig, ax = plt.subplots()
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(title)
            ax.axis("off")

            selected_point = []

            def onclick(event):
                if event.inaxes is None:
                    return
                if len(selected_point) == 0:
                    x, y = event.xdata, event.ydata
                    selected_point.append((round(x), round(y)))
                    ax.plot(x, y, "ro")  # mark selected point
                    fig.canvas.draw()
                    plt.close()

            fig.canvas.mpl_connect("button_press_event", onclick)
            plt.show()

            return selected_point[0] if selected_point else None

        # select the point of left part
        left_point = select_point(left_half, "Select a point on the left half")
        if left_point:
            # Adjust the x-coordinates of the points in the left half to their positions relative to the original image
            points.append((left_point[0], left_point[1]))

        # select the point of right part
        right_point = select_point(right_half, "Select a point on the right half")
        if right_point:
            # Add the x-coordinate of the point in the right half to the width of the left half.
            points.append((right_point[0] + width - 1000, right_point[1]))

        # return the result
        if len(points) == 2:
            return input_folder, output_folder, points
        else:
            print("Error: Less than two points selected.")
            return None


def select_two_points_crop(input_folder, output_folder):
    """
    allow the user to select a point from left side and another point from right side.

    Parameters:
        input_folder (str): the path of the folder of input image
        output_folder (str): the path of the folder of output image

    Returns:
        tuple: (input_folder, output_folder, points)，the form of the points [(x1, y1), (x2, y2)]。
    """
    image_path = None
    points = []
    for straightened_folder in sorted(os.listdir(input_folder)):
        if "straightened" not in straightened_folder:
            continue

        straightened_folder = os.path.join(input_folder, straightened_folder)

        if straightened_folder is None:
            print("Error: No valid straightened image found.")
            return None

        # find the overlay image
        for time_folder in sorted(os.listdir(straightened_folder)):
            if "t00" not in time_folder:
                continue
            time_folder = os.path.join(straightened_folder, time_folder)
            for filename in os.listdir(time_folder):
                if "ch00" in filename.lower() and filename.lower().endswith(
                    (".tif", ".tiff", "jpg")
                ):
                    image_path = os.path.join(time_folder, filename)
                    break

            if image_path is None:
                print("Error: No valid overlay image found.")
                return None

            # load the image
            image = rotate_portrait_to_landscape(
                cv2.imread(image_path, cv2.IMREAD_COLOR)
            )
            if image is None:
                print("Error: Could not open image.")
                return None

            height, width = image.shape[:2]

            # Extract 1000 pixels wide from the leftmost and rightmost parts of the image
            left_half = image[:, :1000]
            right_half = image[:, width - 1000 :]

            def select_point(img, title):
                """Displays the image and allows the user to select a point."""
                fig, ax = plt.subplots()
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                ax.set_title(title)
                ax.axis("off")

                selected_point = []

                def onclick(event):
                    if event.inaxes is None:
                        return
                    if len(selected_point) == 0:
                        x, y = event.xdata, event.ydata
                        selected_point.append((round(x), round(y)))
                        ax.plot(x, y, "ro")  # mark selected point
                        fig.canvas.draw()
                        plt.close()

                fig.canvas.mpl_connect("button_press_event", onclick)
                plt.show()

                return selected_point[0] if selected_point else None

            # select the point of left part
            left_point = select_point(left_half, "Select a point on the left half")
            if left_point:
                # Adjust the x-coordinates of the points in the left half to their positions relative to the original image
                points.append((left_point[0], left_point[1]))

            # select the point of right part
            right_point = select_point(right_half, "Select a point on the right half")
            if right_point:
                # Add the x-coordinate of the point in the right half to the width of the left half.
                points.append((right_point[0] + width - 1000, right_point[1]))

            # return the result
            if len(points) == 2:
                return input_folder, output_folder, points
            else:
                print("Error: Less than two points selected.")
                return None


def batch_crop(input_folder, output_folder, points):
    # Calculate the angle
    pt1, pt2 = points

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    for straightened_folder in sorted(os.listdir(input_folder)):
        if "straightened" not in straightened_folder:
            continue

        straightened_folder = os.path.join(input_folder, straightened_folder)

        for time_folder in sorted(os.listdir(straightened_folder)):
            time = time_folder
            time_folder = os.path.join(straightened_folder, time_folder)
            # Process all images in the folder
            for filename in os.listdir(time_folder):
                image_path = os.path.join(time_folder, filename)
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)

                # Rotate the image to landscape
                landscape_image = rotate_portrait_to_landscape(image)

                # Convert to RGB
                image_rgb = cv2.cvtColor(landscape_image, cv2.COLOR_BGR2RGB)

                # Crop and rotation parameters
                crop_top = min(pt1[1], pt2[1])  # upper boundary
                crop_bottom = max(pt1[1], pt2[1])  # lower boundary
                crop_left = min(pt1[0], pt2[0])  # left boundary
                crop_right = max(pt1[0], pt2[0])  # right boundary

                # Convert cropped image to PIL object and rotate
                image_pil = Image.fromarray(image_rgb)
                cropped_image = image_pil.crop(
                    (crop_left, crop_top, crop_right, crop_bottom)
                )

                # Extract a part of the filename to create subfolder (example: using first character or some part)
                subfolder_name = os.path.join(
                    "cropped image", time
                )  # Modify as needed to extract a specific part
                subfolder_path = os.path.join(output_folder, subfolder_name)

                # Create subfolder if it doesn't exist
                os.makedirs(subfolder_path, exist_ok=True)

                # Save the processed image in the subfolder
                output_path = os.path.join(subfolder_path, f"cropped_{filename}")
                cropped_image.save(output_path)
                print(f"cropped image saved at: {output_path}")


input_folder = r"D:\thesis\processed 7th\R1\Raw"
output_folder = r"D:\thesis\processed 7th\R1\Raw"
# batch_crop(*select_two_points_crop(input_folder,output_folder))
