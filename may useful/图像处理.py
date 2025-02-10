import cv2
import numpy as np
import os

def preprocess_image(image):
    # 将图像转换为灰度
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 应用高斯模糊，减少噪声
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # 二值化图像
    _, binary = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)
    return binary

def detect_pipes(binary):
    # 使用形态学操作识别管道区域
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 5))
    dilated = cv2.dilate(binary, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    pipe_regions = []
    for contour in contours:
        # 获取管道的边界框
        x, y, w, h = cv2.boundingRect(contour)
        pipe_regions.append((x, y, w, h))
    
    # 按位置排序，确保管道顺序
    pipe_regions = sorted(pipe_regions, key=lambda x: x[1])
    return pipe_regions

def count_cells_in_pipe(image, pipe_region):
    x, y, w, h = pipe_region
    pipe_img = image[y:y+h, x:x+w]
    # 预处理以突出显示细胞
    processed_pipe = preprocess_image(pipe_img)
    
    # 检测细胞轮廓
    contours, _ = cv2.findContours(processed_pipe, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cell_count = len(contours)
    
    # 在管道图像上绘制检测结果
    for contour in contours:
        cx, cy, cw, ch = cv2.boundingRect(contour)
        cv2.rectangle(pipe_img, (cx, cy), (cx + cw, cy + ch), (0, 255, 0), 1)
    
    return cell_count, pipe_img

def process_image(image_path):
    # 加载图像
    image = cv2.imread(image_path)
    # 预处理图像
    binary = preprocess_image(image)
    # 检测管道区域
    pipe_regions = detect_pipes(binary)
    
    total_cells = 0
    for idx, pipe_region in enumerate(pipe_regions):
        cell_count, pipe_img = count_cells_in_pipe(image, pipe_region)
        total_cells += cell_count
        print(f"管道 {idx + 1} 中的细胞数量: {cell_count}")
        # 可视化结果，保存单个管道图像
        cv2.imwrite(f"pipe_{idx + 1}_counted.jpg", pipe_img)
    
    print(f"总细胞数量: {total_cells}")

# 设置图像路径
image_path = "path/to/your/image.jpg"
process_image(image_path)
