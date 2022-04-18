# CSCI-635 Final Project
This project is a culmination of the things learned over the spring semester. We will explore 
5 models, each of which have varying levels of success against the 3 datasets we have chosen.
The data is stored in compressed numpy arrays that you can find in the `main/data` folder.

## Project Setup
To begin running the project, you must install several dependencies:

```sh
python -m pip install -r requirements.txt
```

## Running the Project
Running the following command will run all of the augmentation and show the evaluation report.
This will also use the included, pretrained models which are in the `trained_models` folder. To 
generate new models, use some of the below optional arguments.

```
python main/main.py
```

### Optional Command Line Arguments

#### `--training-only`
This can be used to generate the trained models.

```
python main/main.py --training-only
```

#### `--retrain-models`
This argument should be used to retrain the models and overwrite the ones in the  `trained_models` folder.

```
python main/main.py --retrain-models
```

#### `--no-verbose`
To turn off all the extraneous output from the model training, use this argument.

```
python main/main.py --training-only --no-verbose
```

### Making Sense of the Evaluation output
The evaluations step outputs a confusion matrix and an evaluation report for each model with something that looks like this:

```
===== BEGIN RESULTS FOR {MODEL} =====


Confusion Matrix:

===== Confusion Matrix =====
        0       1       2       3       4       5       6       7       8       9       10      11      12      13      14      15      16      17      18      19

0       1922    2       14      4       5       13      11      11      7       8       26      4       1       96      7       1       0       5       2       4
1       0       2199    2       3       0       0       2       6       36      12      4       2       0       0       3       1       0       1       4       0
2       23      8       921     6       11      4       5       13      18      1       3       0       0       0       9       1       6       2       1       0
3       8       12      21      866     0       45      2       18      23      9       2       0       0       1       0       0       3       0       0       0
4       16      6       2       0       891     0       6       1       3       45      2       0       0       0       0       1       1       1       7       0
5       23      15      3       45      3       725     5       4       21      18      6       2       13      1       4       0       0       0       2       2
6       45      4       16      0       11      13      850     0       3       0       1       0       0       0       0       5       1       5       4       0
7       3       34      33      7       6       0       0       1825    2       79      1       1       1       0       2       0       30      0       3       1
8       15      41      8       46      14      34      5       8       884     26      6       11      1       8       1       2       16      15      6       0
9       34      41      1       16      91      14      0       84      5       1843    10      13      1       1       3       0       7       0       6       7
10      11      13      1       0       12      0       0       2       24      48      869     0       3       1       0       0       10      1       4       1
11      4       34      3       1       9       6       0       3       2       33      4       972     38      6       8       6       1       0       23      0
12      3       21      0       1       1       0       0       2       0       38      1       25      895     2       1       0       0       0       10      0
13      34      1       0       1       0       2       0       0       2       3       0       3       0       114     0       1       0       2       0       0
14      13      58      15      0       9       1       1       3       6       14      1       10      3       1       859     0       1       1       4       0
15      3       1       0       0       0       0       1       0       1       1       0       2       0       3       0       158     0       1       8       0
16      2       0       3       4       0       0       0       5       2       3       2       0       0       0       0       4       129     0       0       0
17      9       3       1       1       0       0       2       0       2       0       0       2       0       1       0       2       0       136     0       0
18      16      40      1       1       37      0       8       0       8       4       0       31      4       3       2       5       1       2       995     0
19      11      12      0       0       0       2       0       6       9       27      3       7       3       2       1       0       0       2       0       915

Labels:
    0 = English and Sylheti 0, Arabic 5
    1 = All Languages 1
    2 = English 2
    3 = English 3
    4 = English 4
    5 = English 5
    6 = English 6
    7 = English 7, Arabic 6
    8 = English and Sylheti 8
    9 = All Languages 9
    10 = Arabic 0
    11 = Arabic and Sylheti 2
    12 = Arabic 3
    13 = Sylheti 3
    14 = Arabic 4
    15 = Sylheti 4
    16 = Sylheti 5
    17 = Sylheti 6
    18 = Arabic and Sylheti 7
    19 = Arabic 8

Classification report:
              precision    recall  f1-score   support

           0       0.88      0.90      0.89      2143
           1       0.86      0.97      0.91      2275
           2       0.88      0.89      0.89      1032
           3       0.86      0.86      0.86      1010
           4       0.81      0.91      0.86       982
           5       0.84      0.81      0.83       892
           6       0.95      0.89      0.92       958
           7       0.92      0.90      0.91      2028
           8       0.84      0.77      0.80      1147
           9       0.83      0.85      0.84      2177
          10       0.92      0.87      0.90      1000
          11       0.90      0.84      0.87      1153
          12       0.93      0.90      0.91      1000
          13       0.47      0.70      0.57       163
          14       0.95      0.86      0.90      1000
          15       0.84      0.88      0.86       179
          16       0.63      0.84      0.72       154
          17       0.78      0.86      0.82       159
          18       0.92      0.86      0.89      1158
          19       0.98      0.92      0.95      1000

    accuracy                           0.88     21610
   macro avg       0.85      0.86      0.85     21610
weighted avg       0.88      0.88      0.88     21610

===== END RESULTS FOR {model} =====
```

From this, we can see a couple things. We get precision, recall, and f1-score for each of the labels. We can also see that labels on the side are just numbers. When 
training, the labels had to be generated for the different classes we had (20). The meanings of all of them are also outputted to the console right under the confusion 
matrix.

```
Labels:
    0 = English and Sylheti 0, Arabic 5
    1 = All Languages 1
    2 = English 2
    3 = English 3
    4 = English 4
    5 = English 5
    6 = English 6
    7 = English 7, Arabic 6
    8 = English and Sylheti 8
    9 = All Languages 9
    10 = Arabic 0
    11 = Arabic and Sylheti 2
    12 = Arabic 3
    13 = Sylheti 3
    14 = Arabic 4
    15 = Sylheti 4
    16 = Sylheti 5
    17 = Sylheti 6
    18 = Arabic and Sylheti 7
    19 = Arabic 8
```

Using these, you can see which numbers performed poorly and which ones performed well. The rest of the evaluation metrics are pretty self explanatory.
