import os
from Count_the_overlay import count_overlay
from Count import count_channel
import pandas as pd


def process_image_structure(root_folder):
    """
    解析文件结构并生成计数结果表格。
    """
    data = []

    # 遍历通道文件夹（ch00、ch01等）
    channel_data = {}  # 存储当前图片的所有通道数据
    for channel_name in os.listdir(root_folder):
        channel_name_path = os.path.join(root_folder, channel_name)
        if not os.path.isdir(channel_name_path):
            continue

        # 按下划线分割文件夹名称，检查一下是不是拉直过的
        channel_parts = channel_name.split('_')

        # 示例：假设文件夹名结构为 "ExperimentID_Type_Time"
        if not channel_parts[0] == "straightened":
            continue

        print(f"Processing channel: {channel_parts[-3]}")

        # 遍历通道文件夹内的文件
        for file_name in os.listdir(channel_name_path):
            if file_name.endswith(('.png', '.jpg', '.tif')):
                # 提取试管号和通道号
                parts = file_name.split('_')
                try:
                    tube_number = parts[-1]  # 倒数第二部分表示试管号
                    channel_id = parts[-4]  # 从通道文件夹提取通道名
                except IndexError:
                    print(f"  Skipping file due to unexpected format: {file_name}")
                    continue

                file_path = os.path.join(channel_name_path, file_name)
                print(f"    Processing file: {file_name}")

                # 提取图片名称
                image_name_parts = parts[:5]  # 假设前5部分是图片名称的一部分
                image_name = '_'.join(image_name_parts)  # 拼接图片名称

                # 确保计数只进行在对应的通道上
                if channel_id == "ch00" or channel_id == "ch01":
                    count = count_channel(file_path)
                elif channel_id == "Resize001":
                    count = count_overlay(file_path)
                else:
                    continue  # 跳过不需要处理的通道

                # 将结果存储到 channel_data 中
                if tube_number not in channel_data:
                    channel_data[tube_number] = {"Image Name": image_name, "Tube Number": tube_number}

                channel_data[tube_number][channel_id] = count

    # 将当前图片的数据加入总数据表
    for tube_data in channel_data.values():
        data.append(tube_data)

    # 转换为 DataFrame
    df = pd.DataFrame(data)

    # 根据需要填充缺失值（如填充为0）
    df = df.fillna(0)

    return df

# 主程序
root_folder = r"D:\thesis\coding\data\output_folder"  # 根文件夹路径
output_excel_path = r"D:\thesis\coding\data\output_folder\output.xlsx"  # 输出 Excel 文件路径

df = process_image_structure(root_folder)

# 保存到 Excel
df.to_excel(output_excel_path, index=False)
print(f"Results saved to {output_excel_path}")
