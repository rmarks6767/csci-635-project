import cv2
import sys

num_to_label = {
  0: '0',
  1: '1',
  2: '2-arabic-sylheti',
  3: '3-sylheti',
  4: '4-sylheti',
  5: '5-sylheti',
  6: '6-sylheti',
  7: '7-arabic-sylheti',
  8: '8',
  9: '9',
}

def read_trim_split(image, num_width, num_height, id):
  im_in = cv2.imread(image)

  # Binarize the image to remove extra lines and stuff
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
    # Resize image to match 28x28 mnist data
    im = cv2.resize(image, (28, 28))
    cv2.imwrite(f'./data/sylheti/{id + i}_{num_to_label[i % 10]}.png', im)

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
