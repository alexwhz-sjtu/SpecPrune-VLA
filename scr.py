import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
import re  
import math
def parse_data(filename):
    frames = []
    actions = []
    
    with open(filename, 'r') as f:
        for line in f:
            if 'frame:' in line:
                frame = int(re.findall(r'frame:(\d+)', line)[0])
                frames.append(frame)
            elif 'action:' in line:
                values = re.findall(r'[-]?\d+\.?\d*', line)
                actions.append([float(x) for x in values])
    
    df = pd.DataFrame(actions, columns=['x', 'y', 'z', 'rx', 'ry', 'rz', 'grip'])
    df['frame'] = frames
    df['norm'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
    return df

def find_groups(df):
    """查找每组的起始和结束索引"""
    groups = []
    start_idx = 0
    
    for i in range(1, len(df)):
        if df['frame'].iloc[i] < df['frame'].iloc[i-1]:
            groups.append((start_idx, i-1))
            start_idx = i
    
    groups.append((start_idx, len(df)-1))
    return groups

def plot_velocity_trends(df, groups):
    plt.style.use('default')
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    
    # 添加 xy 平面速度二范数
    df['xy_norm'] = np.sqrt(df['x']**2 + df['y']**2)
    
    main_color = '#1f77b4'  # 基础蓝色
    labels = ['Z Velocity', 'XY-Plane Velocity', 'Total Velocity']
    line_styles = ['--', ':', '-']  # 为三条线设置不同的线型
    
    window = 3  # 平滑窗口
    window_l = 4
    alpha = 0.6  # 统一透明度
    std_scale = 1.5  # 标准差缩放系数
    
    for col, label, ls in zip(['z', 'xy_norm', 'norm'], labels, line_styles):
        # 找到最长的序列长度
        max_length = max(groups[i][1] - groups[i][0] + 1 for i in range(len(groups)))
        all_values = np.zeros((len(groups), max_length))
        all_values[:] = np.nan
        
        # 收集所有组的数据
        for i, (start, end) in enumerate(groups):
            group_data = df.iloc[start:end+1][col].values
            all_values[i, :len(group_data)] = group_data
        
        # 计算均值
        mean_values = np.nanmean(all_values, axis=0)
        
        # 使用更大的窗口计算移动标准差
        rolling_std = pd.Series(mean_values).rolling(window=window, center=True, min_periods=1).std()
        # 使用高斯滤波平滑标准差
        smoothed_std = savgol_filter(rolling_std, window*1+1, 2)
        smoothed_std_l = savgol_filter(rolling_std, window_l*20+1, 3)
        
        # 使用Savitzky-Golay滤波器平滑均值
        if len(mean_values) > window_l:
            smoothed_l = savgol_filter(mean_values, window_l, 2)
            # 确保标准差平滑且不会太小
            min_std = np.mean(smoothed_std_l) * 0.5
            smoothed_std_l = np.maximum(smoothed_std_l, min_std)
        else:
            smoothed_l = mean_values
            smoothed_std_l = rolling_std
        # 使用Savitzky-Golay滤波器平滑均值
        if len(mean_values) > window:
            smoothed = savgol_filter(mean_values, window, 2)
            # 确保标准差平滑且不会太小
            min_std = np.mean(smoothed_std) * 0.5
            smoothed_std = np.maximum(smoothed_std, min_std)
        else:
            smoothed = mean_values
            smoothed_std = rolling_std
        
        frames = np.arange(len(smoothed))
        if col == 'z':
            line_style = '-'
            marker = 'o'
            markersize = 4 
            main_color = "#7dd487ff" 
            color = "#b2eebf86" 
        # elif col == 'xy_norm':
        #     line_style = '-'
        #     main_color = "#d1e30bc9" 
        #     color = "#e4c36eb6" 
        elif col == 'norm':
            line_style = '-'
            marker = 's'
            markersize = 4
            main_color = "#63acd6ff" 
            color = "#abd9f295"  # 深蓝色
        else:
            continue
        
                # 绘制主曲线，使用固定的curve_color
        line = ax.plot(frames, smoothed, 
                      color=main_color,  # 使用统一的曲线颜色
                      linewidth=2, 
                      label=label, 
                      linestyle=line_style, 
                      marker = marker,
                      markersize=markersize,
                      zorder=3)
        
        # 创建渐变阴影效果
        n_lines = 1
        for i in range(n_lines):
            scale = (i + 1) / n_lines * 5
            ax.fill_between(frames, 
                          smoothed_l - smoothed_std_l * scale,
                          smoothed_l + smoothed_std_l * scale,
                          linewidth=0.1,
                          color=color,  # 使用统一的填充颜色
                          alpha=alpha/(i+1.5),
                          zorder=2)
    
    # 设置图表样式

    ax.grid(True, axis='y', color='gray', alpha=0.3, linestyle='--', linewidth=0.7)
    ax.legend(loc='lower left', fontsize=13)
    ax.tick_params(axis='x', which='major', labelsize=8)   # x轴刻度小一点
    ax.tick_params(axis='y', which='major', labelsize=4)   
    
    
    ax.set_xlim(0, len(smoothed) - 1)
    plt.tight_layout(pad=0.6)
    plt.subplots_adjust(top=0.9, bottom=0.18, left=0.12, right=0.95)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig('velocity_trends_combined.png', dpi=300, bbox_inches='tight')
    
    plt.close()

def main():
    filename = '/share/public/wanghanzhen/SpecPrune-VLA/tt.txt'
    df = parse_data(filename)
    groups = find_groups(df)
    plot_velocity_trends(df, groups)

if __name__ == '__main__':
    main()
