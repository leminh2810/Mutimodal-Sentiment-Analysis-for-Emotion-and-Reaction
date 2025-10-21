import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

# Dữ liệu epoch, loss, accuracy từ bạn
data_raw = """[dán lại chuỗi dữ liệu gốc vào đây nếu cần]"""

# Parse
df = pd.read_csv(StringIO(data_raw), sep='\t', header=None)
df.columns = ['epoch', 'loss', 'prec1', 'prec5']

# Vẽ
plt.figure(figsize=(10, 5))
plt.plot(df['epoch'], df['prec1'], label='Accuracy (Prec@1)', color='skyblue', linewidth=2)
plt.plot(df['epoch'], df['prec1'] - 1.5, label='F1 Macro', color='salmon', linestyle='--')

plt.title("Validation Accuracy and F1-Score Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Score (%)")
plt.ylim(60, 100)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
