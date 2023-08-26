def fit(x_train, y_train):
    frequency_table = compute_frequency_table(x_train, y_train)
    rules, error_rates = compute_rules(frequency_table)
    attribute_min_error = find_attribute_with_min_error_rate(error_rates)
    rule = {attribute_min_error: rules[attribute_min_error]}
    return rule


def evaluate(rule, x_test, y_test):
    print("Evaluate test set ...")
    hits = 0
    attribute = list(rule.keys())[0]
    num_rows = x_test[attribute].shape[0]

    for i in range(num_rows):
        value = x_test[attribute].iloc[i]
        expected_class = y_test.iloc[i]
        estimated_class = rule[attribute][value].strip()

        hits += 1 if expected_class == estimated_class else 0

        print('Expected class: {} Estimated class: {} ¿Acierto?: {}'.format(expected_class, estimated_class,
                                                                            expected_class == estimated_class))
    print("Done.")
    return hits



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
    return min(error_rates, key=error_rates.get)


def print_rule(rule):
    attribute = list(rule.keys())[0].strip()
    print('{:^20}|{:^20}|{:^20}'.format('Attribute', 'Value', 'Estimated class'))
    for attribute in rule:
        for value in rule[attribute]:
            print('{:^20}|{:^20}|{:^20}'.format(attribute, value, rule[attribute][value]))
