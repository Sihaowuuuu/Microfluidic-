import math

import cv2
import numpy as np
from PIL import Image

# 加载原始大图和目标沙漏图像
image = cv2.imread(
    r"H:\thesis\Sihao\Raw Image\2 Seed\20240213 red green trap scaled channel_P5 combined TileScan 1_Merged_Resize001_ch02_SV.tif",
    cv2.IMREAD_GRAYSCALE,
)
template = cv2.imread(r"H:\thesis\sand\sand.jpg", cv2.IMREAD_GRAYSCALE)

# 获取模板图像的宽高
h, w = template.shape[:2]

# 使用模板匹配来查找目标图像的位置
res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

# 设置匹配的阈值，越接近1越严格
threshold = 0.7
locations = np.where(res >= threshold)

# 存储所有匹配的中心坐标和图像
matched_centers = []
matched_images = []

# 遍历所有匹配的位置
for pt in zip(*locations[::-1]):
    x, y = pt

    # 计算中心坐标
    center_x = x + w // 2
    center_y = y + h // 2
    matched_centers.append((center_x, center_y))

    # 截取匹配到的沙漏区域
    matched_image = image[y : y + h, x : x + w]
    matched_images.append(matched_image)

# 存储已保存的中心坐标
saved_centers = []

# 显示并打印所有匹配到的沙漏图像及其中心坐标
for i, (center, img) in enumerate(zip(matched_centers, matched_images)):
    should_save = True
    for saved_center in saved_centers:
        # 计算欧几里得距离
        distance = math.sqrt(
            (center[0] - saved_center[0]) ** 2 + (center[1] - saved_center[1]) ** 2
        )
        if distance < 100:
            should_save = False
            break

    # 如果距离足够远，保存图像并记录中心坐标
    if should_save:
        cv2.imwrite(f"hourglass_shape_{i}.png", img)
        saved_centers.append(center)  # 将当前中心坐标添加到已保存坐标列表
        print(f"沙漏图像 {i + 1} 的中心坐标:", center)

# 假设图像是 image，saved_centers 中至少有两个点
pt1 = saved_centers[0]  # 第一个点
pt2 = saved_centers[1]  # 第二个点

# 计算旋转角度
dx = pt2[0] - pt1[0]
dy = pt2[1] - pt1[1]
angle = math.degrees(math.atan2(dy, dx))

# 确定裁剪的宽度范围（从 pt1 到 pt2），保留全高度
x_min = min(pt1[0], pt2[0])
x_max = max(pt1[0], pt2[0])
cropped_image = image[:, x_min:x_max]

# 计算旋转中心为裁剪区域中 pt1 和 pt2 的中点
center = ((pt1[0] - x_min + pt2[0] - x_min) / 2.0, (pt1[1] + pt2[1]) / 2.0)

# 将 numpy.ndarray 转换为 Pillow 图像对象
cropped_image_pil = Image.fromarray(cropped_image)

# 旋转图像
rotated_image = cropped_image_pil.rotate(angle, expand=True)

# 获取旋转后图像的尺寸和中心点
rotated_width, rotated_height = rotated_image.size
center_x, center_y = rotated_width // 2, rotated_height // 2

# 计算裁剪区域的上230 下300
crop_top = max(0, center_y - 230)
crop_bottom = min(rotated_height, center_y + 300)

# 计算左右裁剪的60像素
crop_left = 60
crop_right = rotated_width - 60

# 裁剪旋转后的图像（上下270，左右60）
cropped_rotated_image = rotated_image.crop(
    (crop_left, crop_top, crop_right, crop_bottom)
)

# 显示结果
cropped_rotated_image.show()

"""
# 获取旋转矩阵并应用旋转
rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
rotated_cropped_image = cv2.warpAffine(cropped_image, rotation_matrix, (cropped_image.shape[1], cropped_image.shape[0]))

# 显示结果
cv2.imshow("Cropped Image", cropped_image)
cv2.imshow("Rotated Cropped Image", rotated_cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()"""
