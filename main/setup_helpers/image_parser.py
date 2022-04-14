import cv2
import sys
import numpy as np

def read_trim_split(image, num_width, num_height, id):
  im_in = cv2.imread(image)

  # Binarize the image to remove extra lines
  _, im_bin = cv2.threshold(im_in, 128, 255, cv2.THRESH_BINARY)
  h, w, _ = im_bin.shape

  # Resize the image so it will be split equally
  h = round(h / int(num_height))
  w = round(w / int(num_width))
  im_bin = cv2.resize(im_bin, (w * int(num_width), h * int(num_height)))
  
  # Split the image into the defined pieces
  images = [im_bin[x : x + h, y : y + w] for x in range(0, im_bin.shape[0], h) for y in range(0, im_bin.shape[1], w)]
    
  # Write the images to the sylheti training data file
  for i, image in enumerate(images):
    # 1. Convert to grayscale
    im_gray = 255 - cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Crop to important area
    _, thresh = cv2.threshold(255 - im_gray, 127, 255, cv2.THRESH_BINARY)
    white_pt_coords = np.argwhere(thresh)
    min_y = min(white_pt_coords[:,0])
    min_x = min(white_pt_coords[:,1])
    max_y = max(white_pt_coords[:,0])
    max_x = max(white_pt_coords[:,1])
    crop = im_gray[min_y:max_y,min_x:max_x]

    # 3. Add padding around the cropped image to square it
    w, h = crop.shape
    diff = max(w, h)
    yoff = round((diff-h)/2)
    xoff = round((diff-w)/2)
    new_im = 255 - np.zeros((diff, diff))
    new_im[xoff:xoff+w, yoff:yoff+h] = crop

    # 4. Resize image to match 28x28 mnist data
    im_resize = cv2.resize(im_gray, (28, 28), interpolation = cv2.INTER_AREA)

    # 5. Binarize after scaling
    _, im = cv2.threshold(im_resize, 1, 255, cv2.THRESH_BINARY)

    # 6. Save this new file
    cv2.imwrite(f'./data/sylheti/{id + i}_{i % 10}.png', im)

  return id + len(images)

if __name__ == "__main__":
  # Get the id that we currently have
  f = open("id", "r")
  id = int(f.read())
  f.close()

  # Should have the below 3 args
  if len(sys.argv) != 4:
    print("python image_parser.py [image file] [number of images width] [number of images height]")
  else:
    new_id = read_trim_split(sys.argv[1], sys.argv[2], sys.argv[3], id)
    f = open("id", "w")
    f.write(str(new_id))
    f.close()
