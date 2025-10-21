import pandas as pd
import matplotlib.pyplot as plt

# === Load logs ===
train_log = pd.read_csv('C:\Users\LQM\OneDrive\Máy tính\akaaka_mutil_model_fullcode\results\train_batch0.log', sep='\t')
val_log = pd.read_csv('val0.log', sep='\t')

# === Plot loss ===
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_log['epoch'], train_log['loss'], label='Train Loss')
plt.plot(val_log['epoch'], val_log['loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()
plt.grid(True)

# === Plot accuracy ===
plt.subplot(1, 2, 2)
plt.plot(train_log['epoch'], train_log['prec1'], label='Train Acc')
plt.plot(val_log['epoch'], val_log['prec1'], label='Validation Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy over Epochs')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
