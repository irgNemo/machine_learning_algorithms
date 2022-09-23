import pandas

def train_one_r(dataset_filename, train_sample_size, class_column_name):
    filename = dataset_filename  # Dataset filename
    train_sample_size = train_sample_size  # Training percentage
    class_column_name = class_column_name

    dataset = pandas.read_csv(filename)

    x_train = dataset.sample(frac=train_sample_size)
    y_train = x_train[class_column_name]
    x_train = x_train.drop(columns=class_column_name)

    x_test = dataset.drop(x_train.index)
    y_test = x_test[class_column_name]
    x_test = x_test.drop(columns=[class_column_name])

    frequency_table = compute_frequency_table(x_train, y_train)
    rules, error_rates = compute_rules(frequency_table)
    find_attribute_with_min_error_rate(error_rates)


def compute_frequency_table(x_train, y_train):
    print("Computing frequency table ... ")

    frequency_table = {}
    expected_classes = y_train.unique()

    for column_name in x_train.columns:
        frequency_table[column_name] = {}
        for row_index in range(len(x_train[column_name])):
            value = x_train[column_name].iloc[row_index]
            expected_class = y_train.iloc[row_index]

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
