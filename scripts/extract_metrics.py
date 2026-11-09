import json
from pathlib import Path
import pandas as pd

ROOT = Path(".")
res = {}

files = {
    "ep5": ROOT / "results" / "classifier_eval_ep5.json",
    "ep10": ROOT / "results" / "classifier_eval_ep10.json",
    "ep15": ROOT / "results" / "classifier_eval_ep15.json",
    "baseline": ROOT / "results" / "classifier_eval_summary.json",
}

for k, p in files.items():
    data = json.loads(p.read_text(encoding="utf-8"))
    res[k] = data

logs = {
    "ep5": ROOT / "results" / "training_log_ep5.csv",
    "ep10": ROOT / "results" / "training_log_ep10.csv",
    "ep15": ROOT / "results" / "training_log_ep15.csv",
    "baseline": ROOT / "results" / "training_log_baseline_orig.csv",
}

print("=== LOG SUMMARY ===")
for k, lp in logs.items():
    df = pd.read_csv(lp)
    total_time = df['sec'].sum()
    last_row = df.iloc[-1]
    best_row = df.loc[df['val_loss'].idxmin()]
    print(f"{k}: total_epochs={len(df)}, total_time={total_time:.1f}s")
    print(f"  Last: ep={int(last_row['epoch'])}, tr_loss={last_row['train_loss']:.4f}, tr_acc={last_row['train_acc']:.4f}, va_loss={last_row['val_loss']:.4f}, va_acc={last_row['val_acc']:.4f}, lr={last_row['lr']}")
    print(f"  Best: ep={int(best_row['epoch'])}, tr_loss={best_row['train_loss']:.4f}, tr_acc={best_row['train_acc']:.4f}, va_loss={best_row['val_loss']:.4f}, va_acc={best_row['val_acc']:.4f}, lr={best_row['lr']}")

print("\n=== EVAL METRICS ===")
for k in ["ep5", "ep10", "ep15"]:
    d = res[k]
    lm = d["last_model"]["gt_crops"]
    bm = d["best_model"]["gt_crops"]
    le2e = d["last_model"]["end_to_end"]
    be2e = d["best_model"]["end_to_end"]
    print(f"--- {k} ---")
    print(f"  Last GT: acc={lm['accuracy']*100:.2f}%, macro_r={lm['macro_recall']*100:.2f}%, micro_r={lm['micro_recall']*100:.2f}%, macro_p={lm['macro_precision']*100:.2f}%, macro_f1={lm['macro_f1']*100:.2f}%")
    print(f"  Best GT: acc={bm['accuracy']*100:.2f}%, macro_r={bm['macro_recall']*100:.2f}%, micro_r={bm['micro_recall']*100:.2f}%, macro_p={bm['macro_precision']*100:.2f}%, macro_f1={bm['macro_f1']*100:.2f}%")
    print(f"  Last E2E keys: {list(le2e['cnn'].keys())}")
    print(f"  Last E2E: cnn_map50={le2e['cnn']['mAP@0.5']*100:.2f}%, s1_map50={le2e['stage1']['mAP@0.5']*100:.2f}%")
    print(f"  Best E2E: cnn_map50={be2e['cnn']['mAP@0.5']*100:.2f}%, s1_map50={be2e['stage1']['mAP@0.5']*100:.2f}%")
    print(f"  Last per-class recall:")
    for c, m in lm["per_class"].items():
        print(f"    {c}: R={m['recall']*100:.2f}%, P={m['precision']*100:.2f}%, F1={m['f1']*100:.2f}%, sup={m['support']}")
    print(f"  Best per-class recall:")
    for c, m in bm["per_class"].items():
        print(f"    {c}: R={m['recall']*100:.2f}%, P={m['precision']*100:.2f}%, F1={m['f1']*100:.2f}%, sup={m['support']}")

print("\n--- baseline ---")
bd = res["baseline"]
bgt = bd["gt_crops"]
be2e = bd["end_to_end"]
print(f"  GT: acc={bgt['accuracy']*100:.2f}%")
for c, m in bgt["per_class"].items():
    print(f"    {c}: R={m['recall']*100:.2f}%, P={m['precision']*100:.2f}%, F1={m['f1']*100:.2f}%, sup={m['support']}")
print(f"  E2E: cnn_map50={be2e['cnn']['mAP@0.5']*100:.2f}%, s1_map50={be2e['stage1']['mAP@0.5']*100:.2f}%")
for c in bgt["per_class"]:
    print(f"    {c} AP50: cnn={be2e['cnn']['per_class'][c]['ap']*100:.2f}%, s1={be2e['stage1']['per_class'][c]['ap']*100:.2f}%")
