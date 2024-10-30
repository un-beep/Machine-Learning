import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = r"NILM\Electricity_Data"
# 1. 加载数据集
# 有功功率
P_Data = pd.read_csv(path + r'\Electricity_P.csv', parse_dates=True)

P_Data.astype("int32")

# 2. 将 UNIX 时间戳转换为可读的日期格式
P_Data['timestamp'] = pd.to_datetime(P_Data['UNIX_TS'], unit='s', errors='coerce')
P_Data = P_Data.set_index(['timestamp'])
P_Data.drop('UNIX_TS', axis=1, inplace=True)  # 删除 UNIX 时间戳列

# 3. 使用列名全称替换简写
column_headers = {'WHE': "whole_house", 'RSE':'rental_suite', 'GRE' : 'garage','MHE': 'main_house', 
                  'B1E': 'bedroom_1', 'BME': 'basement', 'CWE': 'clothes_washer','DWE': 'dish_washer', 
                  'EQE': 'equipment', 'FRE': 'furnace', 'HPE': 'heat_pump', 'OFE': 'home_office', 
                  'UTE': 'utility', 'WOE': 'wall_oven', 'B2E': 'bedroom_2', 'CDE': 'clothes_dryer', 
                  'DNE': 'dining_room', 'EBE': 'workbench', 'FGE': 'fridge', 'HTE': 'hot_water', 
                  'OUE': 'outside', 'TVE': 'tv_pvr', 'UNE': 'unmetered'}

P_Data = P_Data.rename(columns=column_headers)

P_Data.info(verbose=True)

# 4. 选取数据以供处理
appliances = ['main_house', 'clothes_washer', 'dish_washer', 'heat_pump', 
              'wall_oven', 'clothes_dryer', 'fridge', 'hot_water', 'tv_pvr']

P_Data = P_Data[appliances]

# print(P_Data.head(n=5))

# Format for converting "datetime" object into string
format_datetime = "%Y-%m-%d %H:%M:%S"

# First timestamp of first day in the dataset as datatype string
first_day_first_index = P_Data.index[0].strftime(format_datetime)

# Last timestamp of first day in the dataset as datatype string
first_day_last_index = P_Data.index[0].replace(hour=23, minute=59, second=0).strftime(format_datetime)

# Title of the plot
title_first_day = "First day energy load curves of AMPDs dataframe"

# Filtering data of first date in the dataframe
first_day = P_Data.loc[first_day_first_index:first_day_last_index]

# Plotting first day
first_day.plot(figsize=(10, 6))

# Declaring y-label
plt.ylabel("Power (W)")

# Displaying the dataframe
plt.show()

# # 3. 数据预处理
# # 检查缺失值
# print("缺失值检查：")
# print(P_Data.isnull().sum())

# # 填补缺失值
# P_Data.fillna(0, inplace=True)  # 或使用 data.dropna() 删除含缺失值的行

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
