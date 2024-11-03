import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

path = r"NILM\Electricity_Data"
# 1. 加载数据集
I_Data = pd.read_csv(path + r'\Electricity_I.csv', parse_dates=True)

# 2. 将 UNIX 时间戳转换为可读的日期格式
I_Data['timestamp'] = pd.to_datetime(I_Data['UNIX_TS'], unit='s', errors='coerce')
I_Data = I_Data.set_index(['timestamp'])
I_Data.drop('UNIX_TS', axis=1, inplace=True)

# 3. 使用列名全称替换简写
column_headers = {
    'WHE': "whole_house", 'RSE': 'rental_suite', 'GRE': 'garage', 'MHE': 'main_house', 
    'B1E': 'bedroom_1', 'BME': 'basement', 'CWE': 'clothes_washer', 'DWE': 'dish_washer', 
    'EQE': 'equipment', 'FRE': 'furnace', 'HPE': 'heat_pump', 'OFE': 'home_office', 
    'UTE': 'utility', 'WOE': 'wall_oven', 'B2E': 'bedroom_2', 'CDE': 'clothes_dryer', 
    'DNE': 'dining_room', 'EBE': 'workbench', 'FGE': 'fridge', 'HTE': 'hot_water', 
    'OUE': 'outside', 'TVE': 'tv_pvr', 'UNE': 'unmetered'
}

I_Data = I_Data.rename(columns=column_headers)

# 3. 数据预处理
I_Data.fillna(0, inplace=True)

# 4. 选取设备数据以供处理
appliances = ['dish_washer', 'clothes_washer', 'furnace', 'fridge']
I_Data = I_Data[appliances]


# 5. 电流量化
mi = 15  # 最大电流量化值
min_distance = 2  # 设置峰值之间的最小距离阈值
min_height = 1e-2  # 设置峰值的最小高度阈值，去掉过小的毛刺
min_current_threshold = 0.1  # 设置忽略的最低电流值

# 统计每个电器电流量化后的统计分布
for appliance in appliances:
    current_values = I_Data[appliance].round(1)  # 保留一位小数
    
    # 只考虑大于阈值的电流值
    filtered_values = current_values[current_values > min_current_threshold]
    value_counts = filtered_values.value_counts().sort_index()  # 统计非零电流值
    
    total_measurements = len(filtered_values)  # 更新总测量数
    pmf = value_counts / total_measurements if total_measurements > 0 else value_counts  # 避免除以零
    
    # 寻找PMF中的峰值
    peaks, properties = find_peaks(pmf.values, height=min_height, distance=min_distance)
    peak_values = pmf.index[peaks]  # 峰值对应的电流值
    
    # 量化状态
    state_probabilities = {}
    state_currents = {}  # 保存状态对应的电流值
    state_index = 0

    # 确定峰值领域
    peak_ranges = []
    for peak in peak_values:
        lower_bound = max(0, peak - 0.2)
        upper_bound = min(mi, peak + 0.2)
        peak_ranges.append((lower_bound, upper_bound))
        
        # 计算该状态的总概率
        total_prob = pmf[(pmf.index >= lower_bound) & (pmf.index <= upper_bound)].sum()
        
        state_probabilities[state_index] = total_prob  # 状态索引与概率对应
        state_currents[state_index] = peak  # 保存电流值
        state_index += 1  # 

    # 确定 OFF 状态的概率
    in_peak_ranges = np.array([False] * len(pmf))
    for lower, upper in peak_ranges:
        in_peak_ranges |= (pmf.index >= lower) & (pmf.index <= upper)

    # 计算 OFF 状态的概率
    off_prob = pmf[~in_peak_ranges].sum()  # 不在峰值领域内的概率
    state_probabilities[state_index] = off_prob  # 将off状态的概率
    state_currents[state_index] = "OFF"  # OFF 状态不对应具体电流
    state_index += 1  # 增加off状态索引

    # 输出状态及其概率
    print(f"Probable load states for {appliance}:")
    for state, prob in state_probabilities.items():
        if state < state_index - 1:  # 判断是否为off状态
            print(f"State {state} (Current: {state_currents[state]:.1f} A): Probability {prob:.4f} (ON state)")
        else:
            print(f"State {state} (Current: {state_currents[state]}): Probability {prob:.4f} (OFF state)")
