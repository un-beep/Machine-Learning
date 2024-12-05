import requests
import os

# 文件下载的URLs
normal_urls = {
    "normal_0": 'https://engineering.case.edu/sites/default/files/97.mat',
    "normal_1": 'https://engineering.case.edu/sites/default/files/98.mat',
    "normal_2": 'https://engineering.case.edu/sites/default/files/99.mat',
    "normal_3": 'https://engineering.case.edu/sites/default/files/100.mat',
}

# 存储路径
path = './data/normal'

# 确保目标文件夹存在
os.makedirs(path, exist_ok=True)

# 下载并保存文件
for normal_url, url in normal_urls.items():
    try:
        response = requests.get(url)
        response.raise_for_status()  # 如果请求失败，会抛出异常
        path_name = os.path.join(path, f"{normal_url}.mat")
        
        # 以二进制模式写入文件
        with open(path_name, 'wb') as f:
            f.write(response.content)
        print(f"文件 {normal_url}.mat 下载成功！")
        
    except requests.exceptions.RequestException as e:
        print(f"下载 {normal_url} 时发生错误: {e}")
