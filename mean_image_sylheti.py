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

imean0=Image.fromarray(mean0.astype(np.uint8))
imean1=Image.fromarray(mean1.astype(np.uint8))
imean2=Image.fromarray(mean2.astype(np.uint8))
imean3=Image.fromarray(mean3.astype(np.uint8))
imean4=Image.fromarray(mean4.astype(np.uint8))
imean5=Image.fromarray(mean5.astype(np.uint8))
imean6=Image.fromarray(mean6.astype(np.uint8))
imean7=Image.fromarray(mean7.astype(np.uint8))
imean8=Image.fromarray(mean8.astype(np.uint8))
imean9=Image.fromarray(mean9.astype(np.uint8))

imean0.save('data\sylheti\mean_images\mean_0_sylethi.png','png')
imean1.save('data\sylheti\mean_images\mean_1_sylethi.png','png')
imean2.save('data\sylheti\mean_images\mean_2_sylethi.png','png')
imean3.save('data\sylheti\mean_images\mean_3_sylethi.png','png')
imean4.save('data\sylheti\mean_images\mean_4_sylethi.png','png')
imean5.save('data\sylheti\mean_images\mean_5_sylethi.png','png')
imean6.save('data\sylheti\mean_images\mean_6_sylethi.png','png')
imean7.save('data\sylheti\mean_images\mean_7_sylethi.png','png')
imean8.save('data\sylheti\mean_images\mean_8_sylethi.png','png')
imean9.save('data\sylheti\mean_images\mean_9_sylethi.png','png')

