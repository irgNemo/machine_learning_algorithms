import pandas as pd
import itertools as it

class NaiveBayes():
    def __init__(self):
        pass

    def fit(self, x_train, y_train):
        likelihood_table = self.compute_likelihood_table(x_train, y_train)

    def predict(self):
        pass

    def compute_likelihood_table(self, x_train, y_train):
        frequency_tables = {}

        for column_name in x_train:
            df_attribute_class = pd.concat([x_train[column_name], y_train], axis=1)
            column_domain = x_train[column_name].unique()
            class_domain = y_train.unique()
            
            df_frequency_table = pd.DataFrame(1, index=list(column_domain), columns=list(class_domain))
            
            for row in df_attribute_class.index:
                val_feature = df_attribute_class.at[row, column_name]
                val_class = df_attribute_class.at[row, y_train.name]
                df_frequency_table.at[val_feature, val_class] += 1

            frequency_tables[column_name] = df_frequency_table

        print(frequency_tables)

