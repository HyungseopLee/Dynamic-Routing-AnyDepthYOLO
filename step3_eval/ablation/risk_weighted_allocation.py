"""Does the risk-weighted objective actually steer compute to safety-critical frames?

Runs each trained router over the KITTI val cache, thresholds A-hat so that exactly
50% of frames take the SUPER path, and reports the SUPER rate separately on frames
that contain a pedestrian/cyclist (loss_sc > 0) and on frames that do not.
"""
import sys, torch, numpy as np
sys.path.insert(0, ".")
from router.router_net import RouterNetwork

dev = "cuda:0"
c = torch.load("results/step2_router/cache/kitti/cache_val_g2x2_both.pt",
               map_location="cpu", weights_only=False)
sc = (c["loss_sc_base"] > 0).numpy()
iv = c["input_base"].to(dev).float()
pv = c["pred_base"].to(dev).float()
pid = torch.zeros(iv.shape[0], dtype=torch.long, device=dev)
print(f"val frames {len(sc)}  with pedestrian/cyclist {sc.sum()} ({sc.mean()*100:.1f}%)")

for w in (1, 2, 5, 10):
    rates_sc, rates_other = [], []
    for s in range(5):
        ck = torch.load(f"results/step2_router/weights/kitti/router_g2x2_both_riskw{w}_s{s}.pt",
                        map_location="cpu", weights_only=False)
        a = ck["args"]
        net = RouterNetwork(group_dim=a["group_dim"], path_dim=a["path_dim"],
                            hidden_dim=a["hidden"], feat=a["feat"], norm=a["norm"],
                            dropout=a["dropout"]).to(dev)
        with torch.no_grad():
            net(iv[:2], pv[:2], pid[:2])
            net.load_state_dict(ck["state_dict"])
            net.eval()
            ah = net.logit(iv, pv, pid).view(-1).cpu().numpy()
        take = ah >= np.median(ah)          # exactly 50% SUPER
        rates_sc.append(take[sc].mean())
        rates_other.append(take[~sc].mean())
    m_sc, m_o = np.mean(rates_sc), np.mean(rates_other)
    print(f"w={w:<3} SUPER rate at 50% budget:  safety-critical frames {m_sc*100:5.1f}% "
          f"(+/-{np.std(rates_sc)*100:.1f})   other frames {m_o*100:5.1f}%   "
          f"gap {(m_sc-m_o)*100:+5.1f} pp")
