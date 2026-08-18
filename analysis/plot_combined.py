'''
python step4_deploy/plot_combined.py \
    --fps_json results/step4_deploy/control/jetson_trtengine_720×1280_b0.85_kp2.0_ki0.33_warmup60_window30_fps.json \
    --eng_json results/step4_deploy/control/jetson_trtengine_720×1280_b0.85_kp2.0_ki0.33_warmup60_window30_energy.json \
    --out jetson_budget_combined_compact.pdf
'''


import json
import argparse
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def trail(x, n):
    """Centered moving average for smoothing realized latency/energy."""
    pad = n // 2
    x_padded = np.pad(x, (pad, n - pad - 1), mode='edge')
    return np.convolve(x_padded, np.ones(n)/n, mode='valid')

SHORT = {
    "daytime": "day", "night": "night", "clear": "clear", 
    "rainy": "rain", "city street": "city", "city": "city", "highway": "hwy"
}
# TITLES = {
#     "night_daytime": "night \u2194 daytime",
#     "city_highway": "city \u2194 highway",
#     "clear_rainy": "clear \u2194 rainy"
# }

def render_compact_combined(path_fps, path_energy, out_path):
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix"
    })
    
    dump_fps = json.loads(Path(path_fps).read_text())
    dump_eng = json.loads(Path(path_energy).read_text())
    
    fams = dump_fps["fam_order"]
    win = dump_fps.get("win", 30)
    
    fps_lo, fps_hi = dump_fps["fps_lo"], dump_fps["fps_hi"]
    ylo_fps, yhi_fps = fps_lo - 2.0, fps_hi + 2.0
    
    e_base, e_super = dump_eng["e_base"], dump_eng["e_super"]
    ylo_eng, yhi_eng = e_base - 20.0, e_super + 40.0
    
    fig, axes = plt.subplots(len(fams), 5, figsize=(22.0, 9.5), 
                             gridspec_kw={'width_ratios': [1, 1, 0.15, 1, 1]}, squeeze=False)
    
    for r in range(len(fams)):
        axes[r][2].set_visible(False)
    
    budgets = [("step", "Step budget"), ("sawtooth", "Sawtooth budget")]
    
    for r, fam in enumerate(fams):
        bounds = dump_fps["families"][fam]["bounds"]
        labels = dump_fps["families"][fam]["labels"]
        
        for b_idx, (bkind, btitle) in enumerate(budgets):
            c_fps = b_idx
            ax_fps = axes[r][c_fps]
            cell_fps = dump_fps["cells"][f"{fam}/{bkind}"]
            tgt_fps = np.asarray(cell_fps["target"])
            sm_fps = 1000.0 / trail(np.asarray(cell_fps["realized"]), win)
            mae_fps = float(np.mean(np.abs(sm_fps - tgt_fps)))
            
            c_eng = b_idx + 3
            ax_eng = axes[r][c_eng]
            cell_eng = dump_eng["cells"][f"{fam}/{bkind}"]
            tgt_eng = np.asarray(cell_eng["target"])
            sm_eng = trail(np.asarray(cell_eng["realized"]), win)
            mae_eng = float(np.mean(np.abs(sm_eng - tgt_eng)))
            
            for ax, tgt, sm, mae, unit in [(ax_fps, tgt_fps, sm_fps, mae_fps, "fps"), 
                                           (ax_eng, tgt_eng, sm_eng, mae_eng, "mJ")]:
                
                for k in range(len(labels)):
                    x0, x1 = bounds[k], bounds[k + 1]
                    ax.axvspan(x0, x1, color="tab:blue" if k % 2 == 0 else "tab:orange", alpha=0.07)
                    
                    ax.text((x0 + x1) / 2, 0.96, SHORT.get(labels[k], labels[k]), 
                            transform=ax.get_xaxis_transform(),
                            ha="center", va="top", fontsize=18, color="0.3")
                    
                    if k > 0: ax.axvline(x0, color="0.6", ls="-", lw=0.6, alpha=0.5)
                
                ax.plot(tgt, color="black", ls="--", lw=1.5, zorder=6)
                ax.plot(sm, color="tab:red", lw=2.0, zorder=5)
                ax.set_xlim(0, len(tgt))
                ax.grid(alpha=0.2, ls="--")
                
                ax.tick_params(labelsize=18)
                
                ax.text(0.02, 0.04, f"MAE={mae:.2f} {unit}" if unit=="fps" else f"MAE={mae:.1f} mJ", 
                        transform=ax.transAxes, va="bottom", ha="left", fontsize=17,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8", alpha=0.85))

            ax_fps.set_ylim(ylo_fps, yhi_fps)
            ax_eng.set_ylim(ylo_eng, yhi_eng)
            
            if r == 0:
                ax_fps.set_title(f"{btitle}", fontsize=20, pad=12)
                ax_eng.set_title(f"{btitle}", fontsize=20, pad=12)
                
            if r == len(fams) - 1:
                ax_fps.set_xlabel("frame", fontsize=19)
                ax_eng.set_xlabel("frame", fontsize=19)
                
            if r < len(fams) - 1:
                ax_fps.set_xticklabels([])
                ax_eng.set_xticklabels([])
            if c_fps == 1: ax_fps.set_yticklabels([])
            if c_eng == 4: ax_eng.set_yticklabels([])


        # axes[r][0].set_ylabel(f"{TITLES.get(fam, fam)}\nFPS", fontsize=19)
        axes[r][0].set_ylabel(f"FPS", fontsize=21)
        axes[r][3].set_ylabel(f"energy (mJ)", fontsize=21) 

    # [수정 1] (a), (b) 캡션 간격 최적화 (본문 겹침 방지 및 적절한 거리 유지)
    axes[-1][0].text(1.0, -0.55, "(a) Target FPS tracking", transform=axes[-1][0].transAxes, 
                     ha="center", fontsize=25)
    axes[-1][3].text(1.0, -0.55, "(b) Target energy tracking", transform=axes[-1][3].transAxes, 
                     ha="center", fontsize=25)

    # [수정 2] 범례를 최상단으로 옮기고 여백 구조 전면 재설정
    # top을 0.88로 줄여 범례가 들어갈 공간 확보, bottom을 0.18로 줄여 낭비되는 공간 축소
    plt.subplots_adjust(left=0.05, right=0.98, wspace=0.1, hspace=0.25, bottom=0.18, top=0.88)
    
    # [수정 3] 범례 최상단(upper center) 배치 및 캔버스 기준 위치 조정
    handles = [Line2D([0], [0], color="black", ls="--", lw=3., label="Target budget"),
               Line2D([0], [0], color="tab:red", lw=3.0, label="Measured")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=24, bbox_to_anchor=(0.5, 0.999))
    
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    print(f"\n[*] Enhanced Compact Figure Saved -> {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps_json", required=True)
    parser.add_argument("--eng_json", required=True)
    parser.add_argument("--out", default="jetson_budget_combined_enhanced.pdf")
    args = parser.parse_args()
    
    render_compact_combined(args.fps_json, args.eng_json, args.out)