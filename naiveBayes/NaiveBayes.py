import pandas as pd
import itertools as it

class NaiveBayes():
    def __init__(self):
        self.frequency_tables = None

    def fit(self, x_train, y_train, laplace_correction=True):
        self.compute_frequency_table(x_train, y_train, laplace_correction)

        for attribute, df in self.frequency_tables.items():
            if attribute[1] == 'categorical':
                for column_name in df.columns:
                   denominator = df[column_name].sum()
                   for idx in df.index:
                       df.at[idx, column_name] = df.at[idx, column_name]/denominator

        return self.frequency_tables

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

    def compute_frequency_table(self, x_train, y_train, laplace_correction):
        self.frequency_tables = {}
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
                self.frequency_tables[column_name] = df_frequency_table
            elif column_name[1] == 'numerical':
                stats_per_class = {}
                for current_class in class_domain:
                    df_filtered_by_class = df_attribute_class.loc[df_attribute_class[y_train.name] == current_class,:]
                    mean = df_filtered_by_class[column_name].mean()
                    std = df_filtered_by_class[column_name].std()
                    stats_per_class[current_class] = {'mean':mean, 'std':std}
                    
                self.frequency_tables[column_name] = stats_per_class

        return self.frequency_tables 
    
    def compute_likelihood_table():
        pass



    def print_frequency_tables(self):
        for attribute in self.frequency_tables:
            output = "{}: \n {} \n".format(attribute, self.frequency_tables[attribute])
            print(output)
