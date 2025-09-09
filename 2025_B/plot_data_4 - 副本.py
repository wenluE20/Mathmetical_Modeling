import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置中文显示和字体大小
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['font.size'] = 18  # 设置全局字体大小

# 读取Excel文件
df = pd.read_excel('附件4.xlsx')

# 提取第一列和第二列数据（假设列名为第一列和第二列，实际使用时可能需要根据实际列名调整）
# 如果需要根据列名访问，可以使用 df['列名'] 替代 df.iloc[:, 0]
x_data = df.iloc[:, 0].values  # 第一列数据
y_data = df.iloc[:, 1].values  # 第二列数据

# 过滤数据在指定区间内的数据
x_min, x_max = 399.6747, 4000.122
y_min, y_max = 0, 100

# 创建筛选条件
#filter_mask = ((x_data >= x_min) & (x_data <= x_max) & 
#               (y_data >= y_min) & (y_data <= y_max))

# 应用筛选条件
#filtered_x = x_data[filter_mask]
#filtered_y = y_data[filter_mask]

# 创建图形，调整尺寸为更适中的大小
plt.figure(figsize=(10, 6))

# 绘制曲线图
plt.plot(x_data, y_data, 'b-', linewidth=3, alpha=0.7)

# 标记特殊点 (399.6747, 0)
special_x = 399.6747
special_y = 0
plt.scatter(special_x, special_y, color='red', s=100, zorder=5, label='特殊点 (399.6747, 0)')
plt.annotate('(399.6747, 0)', xy=(special_x, special_y), xytext=(10, 10),
             textcoords='offset points', arrowprops=dict(arrowstyle='->'),
             fontsize=16, color='red')
plt.legend()

# 设置坐标轴范围
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

# 设置坐标轴标签，增大字体
plt.xlabel('Wavenumber (cm$^{-1}$)', fontsize=16)
plt.ylabel('Absorbance (%)', fontsize=16)

# 添加网格线
plt.grid(True, linestyle='--', alpha=0.7)

# 优化布局
plt.tight_layout()

# 保存图片为png格式
save_path = 'data_4.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')

# 显示图片
plt.show()
