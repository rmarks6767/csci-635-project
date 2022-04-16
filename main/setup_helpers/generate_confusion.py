from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

matrix = np.array()

disp = ConfusionMatrixDisplay(confusion_matrix=matrix)

print(disp)

disp.plot(cmap='Reds')
plt.show()