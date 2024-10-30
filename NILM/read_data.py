import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 加载数据集
# 有功功率
P_Data = pd.read_csv(r'NILM\Electricity_Data\Electricity_P.csv')
# 无功功率
Q_Data = pd.read_csv(r'NILM\Electricity_Data\Electricity_Q.csv')
# 视在功率
S_Data = pd.read_csv(r'NILM\Electricity_Data\Electricity_S.csv')
# 电流
I_Data = pd.read_csv(r'NILM\Electricity_Data\Electricity_I.csv')

# 2. 将 UNIX 时间戳转换为可读的日期格式
P_Data['timestamp'] = pd.to_datetime(P_Data['UNIX_TS'], unit='s')
P_Data.drop('UNIX_TS', axis=1, inplace=True)  # 删除 UNIX 时间戳列

Q_Data['timestamp'] = pd.to_datetime(Q_Data['UNIX_TS'], unit='s')
Q_Data.drop('UNIX_TS', axis=1, inplace=True)  # 删除 UNIX 时间戳列

S_Data['timestamp'] = pd.to_datetime(S_Data['UNIX_TS'], unit='s')
S_Data.drop('UNIX_TS', axis=1, inplace=True)  # 删除 UNIX 时间戳列

I_Data['timestamp'] = pd.to_datetime(I_Data['UNIX_TS'], unit='s')
I_Data.drop('UNIX_TS', axis=1, inplace=True)  # 删除 UNIX 时间戳列



# 3. 数据预处理
# 检查缺失值
print("缺失值检查：")
print(P_Data.isnull().sum())

# 填补缺失值
P_Data.fillna(0, inplace=True)  # 或使用 data.dropna() 删除含缺失值的行

# 4. 数据分析
# 计算总功率消耗

P_Data['total_load'] = P_Data.iloc[:,:-1].sum(axis=1)

# 计算各个电器的平均功率
average_power = P_Data.iloc[:,-1:].mean()
print("\n各电器的平均功率：")
print(average_power)

# 5. 数据可视化
plt.figure(figsize=(12, 6))
plt.plot(P_Data['timestamp'], P_Data['total_load'], label='Total Load', color='blue')
plt.xlabel('Time')
plt.ylabel('Power (W)')
plt.title('Total Power Load Over Time')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
