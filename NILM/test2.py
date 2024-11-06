import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# 1. 加载数据集
path = r"NILM\Electricity_Data"
I_Data = pd.read_csv(path + r'\Electricity_I.csv', parse_dates=True)
P_Data = pd.read_csv(path + r'\Electricity_P.csv', parse_dates=True)

# 2. 处理时间戳
I_Data['timestamp'] = pd.to_datetime(I_Data['UNIX_TS'], unit='s', errors='coerce')
P_Data['timestamp'] = pd.to_datetime(P_Data['UNIX_TS'], unit='s', errors='coerce')
I_Data = I_Data.set_index('timestamp')
P_Data = P_Data.set_index('timestamp')
I_Data.drop('UNIX_TS', axis=1, inplace=True)
P_Data.drop('UNIX_TS', axis=1, inplace=True)

# 3. 替换列名为全称
column_headers = {
    'WHE': "whole_house", 'RSE': 'rental_suite', 'GRE': 'garage', 'MHE': 'main_house', 
    'B1E': 'bedroom_1', 'BME': 'basement', 'CWE': 'clothes_washer', 'DWE': 'dish_washer', 
    'EQE': 'equipment', 'FRE': 'furnace', 'HPE': 'heat_pump', 'OFE': 'home_office', 
    'UTE': 'utility', 'WOE': 'wall_oven', 'B2E': 'bedroom_2', 'CDE': 'clothes_dryer', 
    'DNE': 'dining_room', 'EBE': 'workbench', 'FGE': 'fridge', 'HTE': 'hot_water', 
    'OUE': 'outside', 'TVE': 'tv_pvr', 'UNE': 'unmetered'
}
I_Data = I_Data.rename(columns=column_headers)
P_Data = P_Data.rename(columns=column_headers)

# 4. 数据预处理
I_Data.fillna(0, inplace=True)
P_Data.fillna(0, inplace=True)

# 5. 定义常量
appliances = ['dish_washer', 'tv_pvr', 'furnace', 'fridge']  # 只选择这几个电器
min_distance = 1  # 峰值之间的最小距离
min_height = 1e-2  # 峰值的最小高度
min_current_threshold = 0.1  # 最小电流阈值

# 6. 定义全局变量存储状态范围
global_state_ranges = {}

# 7. 计算电器工作状态
def get_device_states(appliance, current_values, pmf):
    """
    根据PMF识别峰值并将电流范围对应为工作状态。
    """
    # 寻找PMF中的峰值
    peaks, _ = find_peaks(pmf.values, height=min_height, distance=min_distance)
    peak_values = pmf.index[peaks]
    
    # 存储状态概率、状态电流和状态范围
    state_probabilities = {}
    state_currents = {}
    state_ranges = {}
    peak_ranges = []

    # 确定峰值领域
    for peak in peak_values:
        lower_bound = peak - min_distance / 10
        upper_bound = peak + min_distance / 10
        peak_ranges.append((lower_bound, upper_bound))

        # 计算该状态的概率
        total_prob = pmf[(pmf.index >= lower_bound) & (pmf.index <= upper_bound)].sum()
        state_probabilities[peak] = total_prob
        state_currents[peak] = peak
        state_ranges[peak] = (lower_bound, upper_bound)
    
    # 计算OFF状态的概率
    in_peak_ranges = np.array([False] * len(pmf))
    for lower, upper in peak_ranges:
        in_peak_ranges |= (pmf.index >= lower) & (pmf.index <= upper)
    
    off_prob = pmf[~in_peak_ranges].sum()
    state_probabilities['off'] = off_prob
    state_currents['off'] = 0.0
    state_ranges['off'] = (None, None)

    return state_probabilities, state_currents, state_ranges

# 8. 遍历每个电器并计算其工作状态
device_states = {}
for appliance in appliances:
    # 1. 获取电流数据并计算PMF
    current_values = I_Data[appliance].round(1)
    filtered_values = current_values[current_values > min_current_threshold]
    value_counts = filtered_values.value_counts().sort_index()
    total_measurements = len(filtered_values)
    pmf = value_counts / total_measurements if total_measurements > 0 else value_counts

    # 2. 获取状态信息
    state_probabilities, state_currents, state_ranges = get_device_states(appliance, current_values, pmf)
    
    # 3. 输出状态和对应的电流范围、功率平均值
    appliance_states = {}
    for state, prob in state_probabilities.items():
        if state == 'off':
            appliance_states[state] = {'current': state_currents[state], 'power': 0.0}  # OFF状态功率为0
        else:
            # 对应状态下的功率数据求平均值
            appliance_power = P_Data[appliance][(current_values >= state_ranges[state][0]) & 
                                                 (current_values <= state_ranges[state][1])].mean()
            appliance_states[state] = {'current': state_currents[state], 'power': round(appliance_power, 2)}
    
    device_states[appliance] = appliance_states

# 9. 输出结果
for appliance, states in device_states.items():
    print(f"\n{appliance}:")
    for state, details in states.items():
        print(f"  {state} -> Current: {details['current']} A, Power: {details['power']} W")

# 10. 仅保留所选电器的电流和功率数据
I_Data = I_Data[appliances]
P_Data = P_Data[appliances]

# 11. 添加总电流与总功率列，并保留一位小数
I_Data['total_current'] = I_Data.sum(axis=1).round(1)
P_Data['total_power'] = P_Data.sum(axis=1).round(1)

# 12. 添加电流数据的工作状态编码
def get_state_code_for_current(appliance, current_values, state_ranges):
    """
    根据当前电流值为每行电流数据分配工作状态编码
    """
    state_codes = []
    for current in current_values:
        state_code = 'off'  # 默认状态为OFF
        for state, (lower_bound, upper_bound) in state_ranges.items():
            if lower_bound is not None and upper_bound is not None and lower_bound <= current <= upper_bound:
                state_code = f'state_{state}'  # 使用状态编码
                break
        state_codes.append(state_code)
    return state_codes

# 添加电流数据的状态编码列
for appliance in appliances:
    state_codes = get_state_code_for_current(appliance, I_Data[appliance], state_ranges)
    I_Data[f'{appliance}_state_code'] = state_codes  # 添加电流状态编码列

# 13. 使功率数据的状态编码与电流数据一致
for appliance in appliances:
    P_Data[f'{appliance}_state_code'] = I_Data[f'{appliance}_state_code']

# 14. 保存数据
I_Data.to_csv(path + r'\I_data_with_total_and_state.csv')
P_Data.to_csv(path + r'\P_data_with_total_and_state.csv')

# 打印前几行查看结果
print(I_Data.head())
print(P_Data.head())
