from PIL import Image
import os
import re
import logging
paths=os.path.join(re.sub('data_augmentation','',os.path.dirname(__file__)), 'data\sylheti')
def rotation(paths):
    '''Perform a rotation of the images to augment the data'''
    for i in os.listdir(paths):
        try :
            data=Image.open(paths+'\\' +i)
            logging.info('{} open succesfully'.format(i))
            new_data=data.rotate(1)
            filename='\\rotated'+i+'.png'
            new_data.save(paths+filename,'png')
            
        except:
            logging.info('{}is not a valid file it may be a folder so we will just ignore it'.format(i))
        
    
rotation(paths)

