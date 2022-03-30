from PIL import Image
import os
path=r'C:\Users\vsadr\OneDrive\Documents\GitHub\data\sylethi'
for i in os.listdir(path):
    print(i)
    data=Image.open(path+'\\' +i)
    new_data=data.rotate(1)
    filename='augmented'+i
    new_data.save("data\sylethi" + '\\'+ filename,'png')
    break
    


