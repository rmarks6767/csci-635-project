from PIL import Image
import os
data=Image.open(os.path.join(os.path.dirname(__file__)))
new_data=data.rotate(30)
new_data.show()


