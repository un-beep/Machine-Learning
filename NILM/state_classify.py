import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = r"NILM\Electricity_Data"
# 1. 加载数据集
# 电流
I_Data = pd.read_csv(path + r'\Electricity_I.csv', parse_dates=True)

I_Data.astype("int32")

# 2. 将 UNIX 时间戳转换为可读的日期格式
I_Data['timestamp'] = pd.to_datetime(I_Data['UNIX_TS'], unit='s', errors='coerce')
I_Data = I_Data.set_index(['timestamp'])
I_Data.drop('UNIX_TS', axis=1, inplace=True)  # 删除 UNIX 时间戳列

# 3. 使用列名全称替换简写
column_headers = {'WHE': "whole_house", 'RSE':'rental_suite', 'GRE' : 'garage','MHE': 'main_house', 
                  'B1E': 'bedroom_1', 'BME': 'basement', 'CWE': 'clothes_washer','DWE': 'dish_washer', 
                  'EQE': 'equipment', 'FRE': 'furnace', 'HPE': 'heat_pump', 'OFE': 'home_office', 
                  'UTE': 'utility', 'WOE': 'wall_oven', 'B2E': 'bedroom_2', 'CDE': 'clothes_dryer', 
                  'DNE': 'dining_room', 'EBE': 'workbench', 'FGE': 'fridge', 'HTE': 'hot_water', 
                  'OUE': 'outside', 'TVE': 'tv_pvr', 'UNE': 'unmetered'}

I_Data = I_Data.rename(columns=column_headers)

I_Data.info(verbose=True)


# 3. 数据预处理
# 检查缺失值
print("缺失值检查：")
print(I_Data.isnull().sum())

# 填补缺失值
I_Data.fillna(0, inplace=True)  # 或使用 data.dropna() 删除含缺失值的行

# 4. 选取数据以供处理
# appliances = ['main_house', 'clothes_washer', 'dish_washer', 'heat_pump', 
            #   'wall_oven', 'clothes_dryer', 'fridge', 'hot_water', 'tv_pvr']

# P_Data = P_Data[appliances]

print(P_Data.head(n=5))

# # 4. 数据分析
# # 计算总功率消耗

# P_Data['total_load'] = P_Data.iloc[:,:-1].sum(axis=1)

# # 计算各个电器的平均功率
# average_power = P_Data.iloc[:,-1:].mean()
# print("\n各电器的平均功率：")
# print(average_power)

# # 5. 数据可视化
# plt.figure(figsize=(12, 6))
# plt.plot(P_Data['timestamp'], P_Data['total_load'], label='Total Load', color='blue')
# plt.xlabel('Time')
# plt.ylabel('Power (W)')
# plt.title('Total Power Load Over Time')
# plt.legend()
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()
