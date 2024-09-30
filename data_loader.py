import os
import numpy as np
import argparse
from collections import defaultdict

def load_simulation_data(directory, selected_ids=None):
    """
    从指定目录加载模拟数据
    :param directory: 数据文件所在的目录路径
    :param selected_ids: 要加载的车辆ID列表，如果为None则加载所有车辆
    :return: 包含所有车辆数据的字典，键为车辆ID，值为该车辆的所有模拟运行数据列表
    """
    data = defaultdict(list)
    for filename in os.listdir(directory):
        if filename.endswith('.npy'):
            parts = filename.split('_')
            if len(parts) == 2 and parts[0].isdigit() and parts[1].split('.')[0].isdigit():
                vehicle_id = int(parts[0]) + 1  # 文件名中的ID比实际ID小1
                if selected_ids is None or vehicle_id in selected_ids:
                    run_number = int(parts[1].split('.')[0])
                    file_path = os.path.join(directory, filename)
                    run_data = np.load(file_path)
                    
                    # 确保数据按运行次数排序
                    while len(data[vehicle_id]) <= run_number:
                        data[vehicle_id].append(None)
                    data[vehicle_id][run_number] = run_data
    
    # 移除空数据点
    for vehicle_id in data:
        data[vehicle_id] = [run for run in data[vehicle_id] if run is not None]
    
    return data

def print_data_matrix(data):
    """
    打印数据矩阵
    :param data: 包含所有车辆数据的字典
    """
    for vehicle_id, vehicle_runs in data.items():
        print(f"车辆 {vehicle_id} 的数据矩阵:")
        for run_index, run_data in enumerate(vehicle_runs):
            print(f"  运行 {run_index + 1}:")
            print(f"    数据形状: {run_data.shape}")
            print("    数据内容:")
            for layer in range(run_data.shape[2]):
                print(f"      层 {layer}:")
                print(run_data[:,:,layer])
            print()

def main():
    """
    主函数：解析命令行参数，加载数据，并打印数据矩阵
    """
    parser = argparse.ArgumentParser(description="加载并打印车辆模拟数据矩阵")
    parser.add_argument("--directory", type=str, default="E:\\research\\code\\new_pro\\vehicle_data", 
                        help="数据文件所在的目录路径")
    parser.add_argument("--ids", type=int, nargs='+', help="要分析的车辆ID列表，例如：--ids 1 2 3")
    args = parser.parse_args()

    selected_ids = args.ids if args.ids else None
    data = load_simulation_data(args.directory, selected_ids)
    
    if not data:
        print("在指定目录中未找到数据文件或没有匹配的车辆ID。")
        return

    print_data_matrix(data)

if __name__ == "__main__":
    main()