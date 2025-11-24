# plot_training_curves.py
import matplotlib.pyplot as plt
import numpy as np
import os
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# If you saved history as a JSON file named 'history.json'
hist_path = os.path.join(SCRIPT_DIR, "history.json")
if os.path.exists(hist_path):
    h = json.load(open(hist_path))
    acc = h['accuracy']
    val_acc = h['val_accuracy']
    loss = h['loss']
    val_loss = h['val_loss']
else:
    raise SystemExit("save training history as history.json first")

epochs = range(1, len(acc)+1)
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(epochs, acc, label='train_acc')
plt.plot(epochs, val_acc, label='val_acc')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.title('Accuracy')

plt.subplot(1,2,2)
plt.plot(epochs, loss, label='train_loss')
plt.plot(epochs, val_loss, label='val_loss')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.title('Loss')

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "training_curves.png"), dpi=150)
print("Saved training_curves.png")
