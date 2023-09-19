import pandas as pd
import itertools as it
import copy

class NaiveBayes():
    def __init__(self):
        self.frequency_tables = None
        self.likelihood_table = None

    def fit(self, x_train, y_train, laplace_correction=True):
        self.frequency_tables = self.compute_frequency_table(x_train, y_train, laplace_correction)
        frequency_tables_copy = copy.deepcopy(self.frequency_tables)
        self.likelihood_tables = self.compute_likelihood_table(frequency_tables_copy)
        return self.likelihood_tables

    def predict(self, x_train, y_train):
        class_domain = y_train.unique()
        posterior_probability = {}

        for class_value in class_domain:
            for idx in x_train.index:
                for column_name in x_train.columns:
                    value = x_train.loc[idx, column_name]
                    if column_name[1] == 'categorical':
                        pass
                    elif column_name[1] == 'numerical':
                        pass

    def compute_frequency_table(iself, x_train, y_train, laplace_correction):
        frequency_tables = {}
        frequency_init_value = 1.0 if laplace_correction else 0.0

        for column_name in x_train:
            df_attribute_class = pd.concat([x_train[column_name], y_train], axis=1)
            column_domain = x_train[column_name].unique()
            class_domain = y_train.unique()
            
            if column_name[1] == 'categorical':
                df_frequency_table = pd.DataFrame(frequency_init_value, index=list(column_domain), columns=list(class_domain))
                for row in df_attribute_class.index:
                    val_feature = df_attribute_class.at[row, column_name]
                    val_class = df_attribute_class.at[row, y_train.name]
                    df_frequency_table.at[val_feature, val_class] += 1
                frequency_tables[column_name] = df_frequency_table
            elif column_name[1] == 'numerical':
                stats_per_class = {}
                for current_class in class_domain:
                    df_filtered_by_class = df_attribute_class.loc[df_attribute_class[y_train.name] == current_class,:]
                    mean = df_filtered_by_class[column_name].mean()
                    std = df_filtered_by_class[column_name].std()
                    stats_per_class[current_class] = {'mean':mean, 'std':std}
                    
                frequency_tables[column_name] = stats_per_class

        return frequency_tables 
    
    def compute_likelihood_table(self, frequency_tables):
        frequency_tables_copy = frequency_tables.copy()
        for attribute, df in frequency_tables_copy.items():
            if attribute[1] == 'categorical':
                for column_name in df.columns:
                   denominator = df[column_name].sum()
                   for idx in df.index:
                       df.at[idx, column_name] = df.at[idx, column_name]/denominator
        return frequency_tables_copy



    def print_frequency_tables(self):
        for attribute in self.frequency_tables:
            output = "{}: \n {} \n".format(attribute, self.frequency_tables[attribute])
            print(output)
