import os

import imagej

os.environ["JAVA_HOME"] = (
    r"D:\thesis\fiji-win64\Fiji.app\java\win64\zulu8.60.0.21-ca-fx-jdk8.0.322-win_x64\jre"
)
os.environ["PATH"] = os.environ["JAVA_HOME"] + r"bin" + os.pathsep + os.environ["PATH"]

ij = imagej.init(r"D:\thesis\fiji-win64\Fiji.app", mode="interactive")
# 定义宏代码
macro_code = """
open("D:/thesis/coding/data/output_folder/straightened_20240213 red green trap scaled channel_P5 combined TileScan 1_Merged_Resize001_ch00_SV_1024/straightened_20240213 red green trap scaled channel_P5 combined TileScan 1_Merged_Resize001_ch00_SV_1024_28.jpg");
run("8-bit");
setAutoThreshold("Default dark");
setOption("BlackBackground", true);
run("Convert to Mask");
run("Watershed");
run("Analyze Particles...", "display clear"); """
"""
particle_count = nResults
print("Particle Count: " + particle_count);  // 输出粒子计数调试信息
return ""+particle_count;  // 返回粒子数
"""

# 运行宏并获取粒子计数
ij.py.run_macro(macro_code)

# 获取粒子计数 (直接访问 nResults)
particle_count = ij.py.run_macro("return nResults;")  # 通过宏直接获取粒子计数

print(f"检测到的粒子数：{particle_count}")

# 结束 ImageJ 实例
ij.dispose()
