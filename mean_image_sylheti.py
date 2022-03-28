from PIL import Image
import os
import numpy as np
import glob

path=r'C:\Users\vsadr\OneDrive\Documents\GitHub\data\sylheti'
image0,image1,image2,image3,image4,image5,image6,image7,image8,image9=[],[],[],[],[],[],[],[],[],[]
mean0,mean1,mean2,mean3,mean4,mean5,mean6,mean7,mean8,mean9=0,0,0,0,0,0,0,0,0,0
for i in os.listdir(path):
    if '_0' in i:
        image0.append(np.asarray(Image.open(path+'\\' +i)))
    if '_1' in i:
        image1.append(np.asarray(Image.open(path+'\\' +i)))
    if '_2' in i:
        image2.append(np.asarray(Image.open(path+'\\' +i)))
    if '_3' in i:
        image3.append(np.asarray(Image.open(path+'\\' +i)))
    if '_4' in i:
        image4.append(np.asarray(Image.open(path+'\\' +i)))
    if '_5' in i:
        image5.append(np.asarray(Image.open(path+'\\' +i)))
    if '_6' in i:
        image6.append(np.asarray(Image.open(path+'\\' +i)))
    if '_7' in i:
        image7.append(np.asarray(Image.open(path+'\\' +i)))
    if '_8' in i:
        image8.append(np.asarray(Image.open(path+'\\' +i)))
    if '_9' in i:
        image9.append(np.asarray(Image.open(path+'\\' +i)))
for j in image0:
    mean0+=j/len(image0)
for j in image1:
    mean1+=j/len(image1)
for j in image2:
    mean2+=j/len(image2)
for j in image3:
    mean3+=j/len(image3)
for j in image4:
    mean4+=j/len(image4)
for j in image5:
    mean5+=j/len(image5)
for j in image6:
    mean6+=j/len(image6)
for j in image7:
    mean7+=j/len(image7)
for j in image8:
    mean8+=j/len(image8)
for j in image9:
    mean9+=j/len(image9)
mean=[mean0,mean1,mean2,mean3,mean4,mean5,mean6,mean7,mean8,mean9]
# for m in mean:
#    for i in range(len(m)):
#        for j in range(len(m[i])):
#            if m[i][j][0]>180:
#                m[i][j]=[255,255,255]
#            else:
#                m[i][j]=[0,0,0]
                
imean=Image.fromarray(mean9.astype(np.uint8))
imean.show()
