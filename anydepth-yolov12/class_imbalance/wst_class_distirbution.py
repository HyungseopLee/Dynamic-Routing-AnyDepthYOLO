import json
import matplotlib.pyplot as plt
from collections import Counter
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

json_path = '/media/data/bdd100k_yolo/train/attributes.json'
output_pdf = os.path.join(current_dir, 'wst_class_distirbution.pdf')

ATTR_MAPS = {
    "weather": {-1: "undefined", 0: "clear", 1: "rainy", 2: "snowy", 3: "overcast", 4: "foggy", 5: "partly cloudy"},
    "scene": {-1: "undefined", 0: "city street", 1: "highway", 2: "residential", 3: "parking lot", 4: "tunnel", 5: "gas stations"},
    "timeofday": {-1: "undefined", 0: "daytime", 1: "night", 2: "dawn/dusk"}
}

print(f"Loading data from {json_path}...")
with open(json_path, 'r') as f:
    data = json.load(f)

counts = {
    "weather": Counter([v['weather'] for v in data.values()]),
    "scene": Counter([v['scene'] for v in data.values()]),
    "timeofday": Counter([v['timeofday'] for v in data.values()])
}

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('BDD100K Train Set: Context Attribute Distribution', fontsize=16, fontweight='bold')

colors = ['#4C72B0', '#C44E52', '#55A868'] 

for i, attr in enumerate(["weather", "scene", "timeofday"]):
    ax = axes[i]
    attr_counts = counts[attr]
    
    # 🎯 수정된 부분: Counter의 most_common()을 사용하여 빈도수 내림차순으로 (key, count) 튜플 리스트 반환
    sorted_items = attr_counts.most_common()
    labels = [ATTR_MAPS[attr][k] for k, count in sorted_items]
    values = [count for k, count in sorted_items]
    
    bars = ax.bar(labels, values, color=colors[i], edgecolor='black')
    
    ax.set_title(attr.capitalize(), fontsize=14)
    ax.set_ylabel('Number of Images')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(output_pdf, format='pdf', bbox_inches='tight')
print(f"✅ '{output_pdf}' file is created")

'''

python ./class_imbalance/wst_class_distirbution.py

'''