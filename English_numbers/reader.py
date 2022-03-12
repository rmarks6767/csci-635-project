# Link: https://stackoverflow.com/questions/40427435/extract-images-from-idx3-ubyte-file-or-gzip-via-python
from mnist import MNIST
import os

mndata = MNIST(os.path.join(os.path.dirname(__file__), 'samples'))

images, labels = mndata.load_training()
# or
#images, labels = mndata.load_testing()
index = 15  # choose an index ;-)
print(mndata.display(images[index]))