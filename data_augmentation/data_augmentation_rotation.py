from PIL import Image
import os
import re
paths=os.path.join(re.sub('data_augmentation','',os.path.dirname(__file__)), 'data\sylheti')
def rotation(paths):
    for i in os.listdir(paths):
        try :
            data=Image.open(paths+'\\' +i)
        except:
            break
        new_data=data.rotate(1)
        filename='augmented'+i+'.png'
        new_data.save(filename,'png')
    
rotation(paths)

