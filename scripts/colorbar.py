import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.colorbar as colorbar
import os
fig, ax = plt.subplots(figsize=(12, 0.8))  # 拉长宽度，缩短高度
fig.subplots_adjust(left=0.05, right=0.95, bottom=0.65, top=0.99)  # 控制ax位置和大小

cmap = LinearSegmentedColormap.from_list("custom_cmap", [(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)])
norm = Normalize(vmin=0, vmax=1)

cb1 = colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='horizontal')
cb1.set_label('Normalized Flux', fontsize=14)          # 设置colorbar标签字体大小
cb1.ax.tick_params(labelsize=12)            # 设置刻度字体大小
base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, "..", "output")
filename = r"flux_colorbar.png"
file_path = os.path.join(output_dir, filename) 
plt.savefig(file_path, dpi=300)
plt.show()