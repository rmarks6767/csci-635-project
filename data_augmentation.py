from PIL import Image
import os
path=r'C:\Users\vsadr\OneDrive\Documents\GitHub\sylheti-training-data'
for i in os.listdir(path):
    print(i)
    data=Image.open(path+'\\' +i)
    new_data=data.rotate(1)
    filename='augmented'+i
    #new_data.save(filename,'png')
    break
    


