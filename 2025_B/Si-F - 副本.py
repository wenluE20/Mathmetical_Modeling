import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt

# 1. 读取数据
data = pd.read_excel('附件3.xlsx', header=0, names=['波数 (cm-1)', '反射率 (%)'])
x = data['波数 (cm-1)'].values
y = data['反射率 (%)'].values

# 2. 寻找极值点
# 寻找局部极大值 (peak)
max_indices = argrelextrema(y, np.greater, order=100)[0] # order参数控制灵敏度，可根据数据调整
# 寻找局部极小值 (valley)
min_indices = argrelextrema(y, np.less, order=100)[0]

# 确保极值点按X轴顺序排列
max_peaks = sorted(zip(x[max_indices], y[max_indices]), key=lambda i: i[0])
min_valleys = sorted(zip(x[min_indices], y[min_indices]), key=lambda i: i[0])

# 3. & 4. 为每个极小值寻找半宽度点并计算比值
results = []

for i, valley in enumerate(min_valleys):
    v_x, v_y = valley
    
    # 找到当前极小值左右相邻的极大值
    left_peak = None
    right_peak = None
    
    # 寻找左侧极大值 (X坐标小于当前极小值)
    for peak in reversed(max_peaks):
        if peak[0] < v_x:
            left_peak = peak
            break
            
    # 寻找右侧极大值 (X坐标大于当前极小值)
    for peak in max_peaks:
        if peak[0] > v_x:
            right_peak = peak
            break
            
    # 如果找不到任何一侧的极大值，跳过这个极小值
    if left_peak is None or right_peak is None:
        print(f"Warning: Cannot find adjacent peaks for valley at {v_x}. Skipping.")
        continue
        
    l_peak_x, l_peak_y = left_peak
    r_peak_x, r_peak_y = right_peak
    
    # 计算半高度
    half_height = v_y + (min(l_peak_y, r_peak_y) - v_y) / 2
    # 更稳健的做法：使用两侧极大值中较低的那个来计算半高
    # 这适用于非对称峰，确保半高线不会与另一侧的峰相交
    
    # 在当前极小值左右两侧的数据中寻找半高度点
    # 左侧数据 (从left_peak到valley)
    left_region_indices = np.where((x >= l_peak_x) & (x <= v_x))
    left_region_x = x[left_region_indices]
    left_region_y = y[left_region_indices]
    # 插值找到Y值第一次等于half_height的点
    left_half_point_x = np.interp(half_height, left_region_y[::-1], left_region_x[::-1])
    
    # 右侧数据 (从valley到right_peak)
    right_region_indices = np.where((x >= v_x) & (x <= r_peak_x))
    right_region_x = x[right_region_indices]
    right_region_y = y[right_region_indices]
    right_half_point_x = np.interp(half_height, right_region_y, right_region_x)
    
    # 计算半宽度 (FWHM)
    fwhm = right_half_point_x - left_half_point_x
    
    # 计算两侧极大值的X距离
    peak_distance = r_peak_x - l_peak_x
    
    # 计算比值
    ratio = peak_distance / fwhm
    
    # 存储结果
    results.append({
        'Valley_X': v_x,
        'Left_Half_Point_X': left_half_point_x,
        'Right_Half_Point_X': right_half_point_x,
        'FWHM': fwhm,
        'Left_Peak_X': l_peak_x,
        'Right_Peak_X': r_peak_x,
        'Peak_Distance': peak_distance,
        'Ratio': ratio
    })

# 将结果转换为DataFrame以便分析
df_results = pd.DataFrame(results)

# 5. 结果可视化
# 绘制原始数据及极值点

plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Reflectance Spectrum', color='lightgray')
# 标记极值点
plt.scatter(x[max_indices], y[max_indices], color='green', marker='^', s=50, label='Maxima (Peaks)')
plt.scatter(x[min_indices], y[min_indices], color='red', marker='v', s=50, label='Minima (Valleys)')
# 标记半宽度点和线
for res in results:
    v_x = res['Valley_X']
    lh_x = res['Left_Half_Point_X']
    rh_x = res['Right_Half_Point_X']
    half_height_val = np.interp(lh_x, x, y) # 获取半高度点的Y值用于绘图
    plt.hlines(y=half_height_val, xmin=lh_x, xmax=rh_x, colors='blue', linestyles='dashed')
    plt.scatter([lh_x, rh_x], [half_height_val, half_height_val], color='blue', marker='o', s=30, label='Half-Width Points' if 'Half-Width Points' not in plt.gca().get_legend_handles_labels()[1] else "")
plt.xlabel('Wavenumber (cm⁻¹)')
plt.ylabel('Reflectance (%)')
plt.title('Reflectance Spectrum with Extrema and Half-Width Analysis')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('极值与半宽度分析.png', dpi=300, bbox_inches='tight')
plt.show()

# 绘制比值变化图


plt.figure(figsize=(10, 5))
plt.plot(df_results['Valley_X'], df_results['Ratio'], 'o-', color='orange')
plt.xlabel('Wavenumber of Valley (cm⁻¹)')
plt.ylabel('Ratio (Peak Distance)')
plt.title('Change of Ratio (Peak Distance) for Each Minimum')
plt.grid(True, alpha=0.3)
# 为每个点标注具体值
for i, row in df_results.iterrows():
    plt.annotate(f"{row['Ratio']:.2f}", (row['Valley_X'], row['Ratio']), textcoords="offset points", xytext=(0,10), ha='center', fontsize=12)

# 添加y=2.5的水平直线
y_line = 2.5
plt.axhline(y=y_line, color='blue', linestyle='--', label='y=2.5')

# 查找交点（线性插值法）
vx = df_results['Valley_X'].values
vy = df_results['Ratio'].values
cross_points = []
for i in range(1, len(vx)):
    if (vy[i-1] - y_line) * (vy[i] - y_line) < 0:
        # 线性插值求交点x
        x_cross = vx[i-1] + (vx[i] - vx[i-1]) * (y_line - vy[i-1]) / (vy[i] - vy[i-1])
        cross_points.append((x_cross, y_line))
        plt.scatter(x_cross, y_line, color='red', zorder=5)
        plt.annotate(
    f"({x_cross:.1f}, {y_line})",
    (x_cross, y_line),
    fontsize=12,
    color='red'
)

plt.tight_layout()
plt.savefig('比值变化.png', dpi=300, bbox_inches='tight')
plt.show()

# 打印结果表格
print(df_results.to_string())
# 保存结果表格为Excel
df_results.to_excel('极值与半宽度分析结果.xlsx', index=False)
print('结果表格已保存为 极值与半宽度分析结果.xlsx')