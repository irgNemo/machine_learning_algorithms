#! /usr/bin/env python
import OneR
import pandas


def main():
    filename = "../datasets/golf-dataset-categorical.csv"  # Dataset filename
    train_sample_size = .8  # Training percentage
    class_column_name = 'Play'

    dataset = pandas.read_csv(filename)

    x_train = dataset.sample(frac=train_sample_size)
    y_train = x_train[class_column_name]
    x_train = x_train.drop(columns=class_column_name)

    x_test = dataset.drop(x_train.index)
    y_test = x_test[class_column_name]
    x_test = x_test.drop(columns=[class_column_name])

    model = OneR.fit(x_train, y_train)
    OneR.print_rule(model)

    hits = OneR.evaluate(model, x_test, y_test)
    test_size = len(y_test)

    print('Hits {} --- Accuracy: {}'.format(hits, hits/test_size))


if __name__ == '__main__':
    main()

