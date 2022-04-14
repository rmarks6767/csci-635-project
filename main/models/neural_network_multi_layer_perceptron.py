import os.path
from sklearn.neural_network import MLPClassifier
from mnist import MNIST

from main.data.data_loader import load_all_data


def main():
    # PLAYING AROUND With ENGLISH DATA
    (train_data, train_label), (test_data, test_label) = load_all_data(True)
    # Insight into choosing hidden layers and their nodes:
    # https://www.linkedin.com/pulse/choosing-number-hidden-layers-neurons-neural-networks-sachdev
    # 3 hidden layers
    clf = MLPClassifier(hidden_layer_sizes=(588, 392, 196),
                        activation='relu',
                        solver='sgd',
                        alpha=0.0001,
                        batch_size='auto',
                        learning_rate='constant',
                        learning_rate_init=0.001,
                        random_state=1)
    clf.fit(train_data, train_label)
    prediction = clf.predict(test_data)
    idx = 0
    correct = 0
    total = len(english_test_label)
    while idx < total:
        if prediction[idx] == test_label[idx]:
            correct += 1
        idx += 1
    print(correct/total)


if __name__ == "__main__":
    main()