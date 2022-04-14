import numpy as np
import skimage.io
import matplotlib.pyplot as plt
import os
paths=os.path.join(os.path.dirname(__file__).replace('setup_helpers',''), 'data\sylheti\mean_images\mean_3_sylethi.png')
image=skimage.io.imread(paths)
histogram, bin_edges = np.histogram(image, bins=256, range=(0, 256))
plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("grayscale value")
plt.ylabel("pixel count")
plt.xlim([0.0, 256.0])
plt.plot(bin_edges[0:-1], histogram)
plt.show()