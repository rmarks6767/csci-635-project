'''
  
  We will use this file to create versions of all of our models and easily compare the
  results of each of them in order to pick the best one.

''' 
from models.svm import SVM
from models.cnn import CNN

def main():
  # Create all of the models that we have
  svm = SVM()
  cnn = CNN()

  # Train them all (if they do not exist already)
  svm.train()
  cnn.train()

  # Analyze the results of the training (maybe use some data for cross validation)

if __name__ == "__main__":
    main()