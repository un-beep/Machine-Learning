import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 加载数据集
data = pd.read_csv(r'NILM\Electricity_P.csv')

# 2. 将 UNIX 时间戳转换为可读的日期格式
data['timestamp'] = pd.to_datetime(data['UNIX_TS'], unit='s')
data.drop('UNIX_TS', axis=1, inplace=True)  # 删除 UNIX 时间戳列

# 3. 数据预处理
# 检查缺失值
print("缺失值检查：")
print(data.isnull().sum())

# 填补缺失值
data.fillna(0, inplace=True)  # 或使用 data.dropna() 删除含缺失值的行

# 4. 数据分析
# 计算总功率消耗
temp = data
temp.drop('timestamp', axis=1, inplace=True)  # 删除 UNIX 时间戳列
data['total_load'] = temp.sum(axis=1)

# 计算各个电器的平均功率
average_power = temp.mean()
print("\n各电器的平均功率：")
print(average_power)

# 5. 数据可视化
plt.figure(figsize=(12, 6))
plt.plot( data['total_load'], label='Total Load', color='blue')
plt.xlabel('Time')
plt.ylabel('Power (W)')
plt.title('Total Power Load Over Time')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
