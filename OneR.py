import pandas

def train_one_r(dataset_filename, train_sample_size, class_column_name):
    filename = dataset_filename  # Dataset filename
    train_sample_size = train_sample_size  # Training percentage
    test_sample_size = 1 - train_sample_size
    class_column_name = class_column_name

    dataset = pandas.read_csv(filename)

    xTrain = dataset.sample(frac=train_sample_size)
    yTrain = xTrain[class_column_name]
    xTrain = xTrain.drop(columns=class_column_name)

    xTest = dataset.drop(xTrain.index)
    yTest = xTest[class_column_name]
    xTest = xTest.drop(columns=[class_column_name])

    frequency_table = compute_frequency_table(xTrain, yTrain)
    rules, error_rates = compute_rules(frequency_table)
    find_attribute_with_min_error_rate(error_rates)


def compute_frequency_table(xTrain, yTrain):
    print("Computing frequency table ... ")

    frequency_table = {}
    expected_classes = yTrain.unique()

    for column_name in xTrain.columns:
        frequency_table[column_name] = {}
        for row_index in range(len(xTrain[column_name])):
            value = xTrain[column_name].iloc[row_index]
            expected_class = yTrain.iloc[row_index]

            value = value.strip() if type(value) is str else value

            if value not in frequency_table[column_name].keys():
                frequency_table[column_name][value] = {}
                for class_name in expected_classes:
                    frequency_table[column_name][value][class_name] = 0  # Initialization with laplace correction

            frequency_table[column_name][value][expected_class] += 1

    print("Done.")
    return frequency_table


def compute_rules(frequency_table):
    print("Finding rule with minimum error rate ... ")
    rules = {}
    total_error_rate = {}
    for attribute in frequency_table:
        total_error_numerator = 0
        total_error_denominator = 0
        for value in frequency_table[attribute]:
            max_frequency = None
            max_target = None
            min_frequency_sum = 0
            frequency_per_value = 0
            for current_target in frequency_table[attribute][value]:
                current_frequency = frequency_table[attribute][value][current_target]
                frequency_per_value += current_frequency

                if max_frequency is None:
                    max_frequency = current_frequency
                    max_target = current_target
                elif current_frequency > max_frequency:
                    max_frequency = current_frequency
                    max_target = current_target

            total_error_numerator += frequency_per_value - max_frequency
            total_error_denominator += frequency_per_value

            if attribute not in rules.keys():
                rules[attribute] = {}
            if value not in rules[attribute].keys():
                rules[attribute][value] = {}

            rules[attribute][value] = max_target

        total_error_rate[attribute] = total_error_numerator / total_error_denominator

    print('Done.')

    return rules, total_error_rate


def find_attribute_with_min_error_rate(error_rates):
    for attribute in error_rates:
        print(attribute)