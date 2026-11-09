import json
import csv
from pathlib import Path

CLASSES = ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]

def analyze_confusion(cm, classes=CLASSES):
    misclassifications = []
    for i, true_cls in enumerate(classes):
        for j, pred_cls in enumerate(classes):
            if i != j and cm[i][j] > 0:
                misclassifications.append((true_cls, pred_cls, cm[i][j]))
    return misclassifications

def inspect():
    # Baseline
    b_eval = json.load(open('results/classifier_eval_summary.json'))
    b_log = list(csv.DictReader(open('results/training_log.csv')))
    b_best_ep = min(range(len(b_log)), key=lambda i: float(b_log[i]['val_loss'])) + 1
    b_last_ep = len(b_log)
    print(f"Baseline: last ep={b_last_ep}, best ep={b_best_ep}")
    print(f"  Last train_loss={b_log[-1]['train_loss']}, val_loss={b_log[-1]['val_loss']}, val_acc={b_log[-1]['val_acc']}")
    print(f"  Best train_loss={b_log[b_best_ep-1]['train_loss']}, val_loss={b_log[b_best_ep-1]['val_loss']}, val_acc={b_log[b_best_ep-1]['val_acc']}")

    for ep in ['ep5', 'ep10', 'ep15']:
        p_eval = json.load(open(f'results/classifier_eval_{ep}.json'))
        p_log = list(csv.DictReader(open(f'results/training_log_{ep}.csv')))
        best_ep = min(range(len(p_log)), key=lambda i: float(p_log[i]['val_loss'])) + 1
        last_ep = len(p_log)
        print(f"\n{ep.upper()}: last ep={last_ep}, best ep={best_ep}")
        print(f"  Last: train_loss={p_log[-1]['train_loss']}, val_loss={p_log[-1]['val_loss']}, val_acc={p_log[-1]['val_acc']}")
        print(f"  Best: train_loss={p_log[best_ep-1]['train_loss']}, val_loss={p_log[best_ep-1]['val_loss']}, val_acc={p_log[best_ep-1]['val_acc']}")
        
        last_m = p_eval['last_model']
        best_m = p_eval['best_model']
        
        cm_last = last_m['gt_crops']['confusion_matrix']
        cm_best = best_m['gt_crops']['confusion_matrix']
        
        print("  Last misclassifications:", analyze_confusion(cm_last))
        print("  Best misclassifications:", analyze_confusion(cm_best))

if __name__ == '__main__':
    inspect()

