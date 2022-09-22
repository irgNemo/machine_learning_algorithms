import OneR


def main():
    dataset_filename = "datasets/golf-dataset-categorical.csv"

    OneR.train_one_r(dataset_filename, 1, 'Play')


if __name__ == '__main__':
    main()

