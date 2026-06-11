"""Online router training WITH data augmentation (overfitting probe).

Unlike train_policy.py (which trains on a frozen offline feature cache and thus
sees each image as one fixed vector), this trains the router end-to-end against
the frozen detector inside the loop: every step loads augmented images
(flip / scale / translate / hsv; NO mosaic/mixup so the per-image advantage
A = L_base - L_super stays well defined), runs both paths through the frozen
detector to get features + per-image losses, and updates only the router.

This is a diagnostic to see whether augmentation lifts the validation
correlation ceiling that caching cannot exceed. Validation uses the existing
deterministic cache (cache_val_g2.pt), identical to train_policy's val_corr.

    conda run -n yolov12 python -m method02_advantage_regress_tinyConv.train_policy_online \
        --weight runs/kitti/detect/anydepth-yolov12s/train/weights/best.pt \
        --data ultralytics/cfg/datasets/kitti.yaml --epochs 100 --seed 0
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from ultralytics import YOLO
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.loss import v8DetectionLoss

from method02_advantage_regress_tinyConv.build_cache import (
    num_skippable, forward_path, grids, per_image_losses, parse_grid,
)
from method02_advantage_regress_tinyConv.feature_tap import (
    STATE_LAYERS, INPUT_LEVEL_LAYERS,
)
from method02_advantage_regress_tinyConv.policy_net import PolicyNetwork

BASE = Path(__file__).resolve().parent / "outputs"


@torch.no_grad()
def val_corr(net, cache):
    net.eval()
    A = (cache["loss_base"].view(-1) - cache["loss_super"].view(-1))
    pid = torch.zeros(A.shape[0], dtype=torch.long, device=A.device)
    ah = net.logit(cache["input_base"], None, pid).view(-1)
    a = ah - ah.mean(); b = A - A.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", default="runs/kitti/detect/anydepth-yolov12s/train/weights/best.pt")
    ap.add_argument("--data", default="ultralytics/cfg/datasets/kitti.yaml")
    ap.add_argument("--dataset", default="kitti")
    ap.add_argument("--val_cache", default=None)
    ap.add_argument("--flops", default=None)
    ap.add_argument("--imgsz", type=int, nargs=2, default=[384, 1248])
    ap.add_argument("--grid", default="2")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--group_dim", type=int, default=64)
    ap.add_argument("--prev_p", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    base = BASE / args.dataset
    if args.val_cache is None: args.val_cache = str(base / "cache_val_g2.pt")
    if args.flops is None:     args.flops = str(base / "flops_table.json")
    if args.out is None:       args.out = str(base / "policy_online.pt")

    torch.manual_seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"

    # frozen detector + feature hooks (backbone input-level layers)
    yolo = YOLO(args.weight, task="detect")
    model = yolo.model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.save = sorted(set(list(model.save) + list(STATE_LAYERS)))
    model.num_skippable_layers = num_skippable(model)
    N = model.num_skippable_layers
    detect = model.model[-1]
    criterion = v8DetectionLoss(model)
    if isinstance(getattr(criterion, "hyp", None), dict) or not hasattr(criterion.hyp, "box"):
        criterion.hyp = DEFAULT_CFG
    captured = {}
    handles = [model.model[l].register_forward_hook(
        lambda m, i, o, l=l: captured.__setitem__(l, o)) for l in STATE_LAYERS]

    # augmented TRAIN dataloader: flip/scale/translate/hsv, NO mosaic/mixup so the
    # per-image advantage label stays defined; rect letterbox matches the val cache.
    data = check_det_dataset(args.data)
    cfg = DEFAULT_CFG
    cfg.imgsz = max(args.imgsz)
    cfg.rect = True
    cfg.mosaic = 0.0
    cfg.mixup = 0.0
    cfg.copy_paste = 0.0
    dataset = build_yolo_dataset(cfg, data["train"], args.batch, data,
                                 mode="train", rect=True, stride=int(model.stride.max()))
    loader = build_dataloader(dataset, args.batch, workers=4, shuffle=False)

    val_cache = torch.load(args.val_cache, map_location=device, weights_only=False)
    val_cache["input_base"] = val_cache["input_base"].to(device).float()
    val_cache["loss_base"] = val_cache["loss_base"].to(device)
    val_cache["loss_super"] = val_cache["loss_super"].to(device)

    skip_base, skip_super = [True] * N, [False] * N
    G = parse_grid(args.grid)

    net = PolicyNetwork(group_dim=args.group_dim, path_dim=8, hidden_dim=args.hidden,
                        feat="input", norm="batch", dropout=args.dropout).to(device)

    def features_and_losses(batch):
        img = batch["img"].to(device).float() / 255.0
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        b["img"] = img
        feats, losses = {}, {}
        for skip, tag in ((skip_base, "base"), (skip_super, "super")):
            captured.clear()
            preds = forward_path(model, img, skip, detect)
            feats[tag] = grids(captured, INPUT_LEVEL_LAYERS, G)
            losses[tag] = per_image_losses(criterion, preds, b)
        return feats, losses

    # materialise lazy layers with one real batch, then build the optimizer
    first = next(iter(loader))
    with torch.no_grad():
        f0, _ = features_and_losses(first)
        net(f0["base"][:2], None, torch.zeros(2, dtype=torch.long, device=device))
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_corr, best_state, best_ep = -1.0, None, -1
    for ep in range(args.epochs):
        net.train()
        run_acc, run_tr_corr, nb = 0.0, 0.0, 0
        for batch in loader:
            feats, losses = features_and_losses(batch)
            B = losses["base"].shape[0]
            A = (losses["base"] - losses["super"]).detach()
            prev = (torch.rand(B, device=device) < args.prev_p).long()
            inp = feats["base"].clone()
            m = prev == 1
            if m.any():
                inp[m] = feats["super"][m]
            out = net.logit(inp, None, prev).view(-1)
            loss = torch.nn.functional.mse_loss(out, A)
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                a = out - out.mean(); b2 = A - A.mean()
                tr_c = float((a * b2).sum() / (a.norm() * b2.norm() + 1e-8))
            run_acc += float(loss) * B; run_tr_corr += tr_c * B; nb += B
        vc = val_corr(net, val_cache)
        tag = ""
        if vc > best_corr:
            best_corr, best_ep = vc, ep
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            tag = "  *best*"
        print(f"ep {ep:3d} | train mse {run_acc/nb:.4f} train_corr {run_tr_corr/nb:.3f} "
              f"|| val_corr {vc:.4f}{tag}")

    for h in handles:
        h.remove()
    torch.save({"state_dict": best_state, "best_corr": best_corr, "best_epoch": best_ep,
                "args": vars(args)}, args.out)
    print(f"[*] best val_corr={best_corr:.5f} @ epoch {best_ep} (of {args.epochs}) -> {args.out}")


if __name__ == "__main__":
    main()
