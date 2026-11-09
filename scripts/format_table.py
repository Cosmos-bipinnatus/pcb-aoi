import json
import csv

CLASSES = ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]

def get_row(ep_key, last_data, best_data, last_log, best_log, best_ep_num, is_baseline=False):
    def fmt_num(val_last, val_best, is_pct=False, decimals=2, loss_dec=4):
        if is_pct:
            s_last = f"{val_last * 100:.{decimals}f}%"
            s_best = f"{val_best * 100:.{decimals}f}%"
        else:
            s_last = f"{val_last:.{loss_dec}f}"
            s_best = f"{val_best:.{loss_dec}f}"
        if s_last == s_best:
            return s_last
        return f"{s_last} (Best@Ep{best_ep_num}: {s_best})"

    # loss and acc from training log
    tl_last = float(last_log["train_loss"])
    tl_best = float(best_log["train_loss"])
    train_loss = fmt_num(tl_last, tl_best, is_pct=False, loss_dec=4)

    vl_last = float(last_log["val_loss"])
    vl_best = float(best_log["val_loss"])
    val_loss = fmt_num(vl_last, vl_best, is_pct=False, loss_dec=4)

    va_last = float(last_log["val_acc"])
    va_best = float(best_log["val_acc"])
    val_acc = fmt_num(va_last, va_best, is_pct=True, decimals=2)

    # test gt metrics
    mr_last = last_data["gt_crops"]["macro_recall"]
    mr_best = best_data["gt_crops"]["macro_recall"]
    macro_recall = fmt_num(mr_last, mr_best, is_pct=True, decimals=2)

    acc_last = last_data["gt_crops"]["accuracy"]
    acc_best = best_data["gt_crops"]["accuracy"]
    micro_recall = fmt_num(acc_last, acc_best, is_pct=True, decimals=2)

    # per-class recalls
    recalls = {}
    for c in CLASSES:
        rc_last = last_data["gt_crops"]["per_class"][c]["recall"]
        rc_best = best_data["gt_crops"]["per_class"][c]["recall"]
        recalls[c] = fmt_num(rc_last, rc_best, is_pct=True, decimals=2)

    # e2e cnn mAP
    e2e_last = last_data["end_to_end"]["cnn"]["mAP@0.5"]
    e2e_best = best_data["end_to_end"]["cnn"]["mAP@0.5"]
    e2e_map = fmt_num(e2e_last, e2e_best, is_pct=True, decimals=2)

    return {
        "group": ep_key,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "macro_recall": macro_recall,
        "micro_recall": micro_recall,
        "recalls": recalls,
        "e2e_map": e2e_map,
    }

print("Script template ready.")

