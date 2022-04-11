import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from read_all_data import read_all_data

def run_training():
  # Get the images from the directory
  images, labels = read_all_data()

  print('MAKING DATA FRAME')
  df = pd.DataFrame(images)
  df['Target'] = labels
  x = df.iloc[:, :-1]
  y = df.iloc[:, -1]

  param_grid = {'C': [0.1, 1, 10, 100], 'kernel': ['rbf', 'poly']}

  print('CREATING SVC AND PERFORMING GRID SEARCH')
  svc = svm.SVC()
  model = GridSearchCV(svc, param_grid)

  print('SPLITTING DATA')
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 77, stratify = y)

  print('FITTING THE MODEL')
  model.fit(x_train, y_train)

  print('PREDICTING THE MODEL')
  y_pred = model.predict(x_test)

  print('The predicted data is:')
  print(y_pred)
  print('The actual data is:')
  print(np.array(y_test))
  print(f'The model is {accuracy_score(y_pred, y_test) * 100}% accurate')

run_training()