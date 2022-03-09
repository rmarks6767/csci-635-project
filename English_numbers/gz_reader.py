import gzip
file1=r"C:\Users\vsadr\OneDrive\Documents\GitHub\arabic_numbers\t10k_images_idx3_ubyte.gz"
print (gzip.open(file1,'rb').read()[1:100])