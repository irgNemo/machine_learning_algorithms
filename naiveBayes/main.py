#!/usr/bin/env python

import pandas as pd
import argparse
import naive_bayes as nb

def main():
    args = parameters_parsing().parse_args()
    df = pd.read_csv(args.input_file)
    train_set, test_set = split_dataset(df=df, train_size=0.7)
    x_train = train_set.iloc[:,0:-1]
    y_train = train_set.iloc[:,-1]
    nb.naive_bayes(x_train, y_train)

def split_dataset(df:pd.DataFrame, train_size:int)->tuple:
    train_set = df.sample(frac=train_size)
    test_set = df.drop(index=train_set.index)
    return train_set, test_set
    
def parameters_parsing()->argparse.Namespace:
    parser = argparse.ArgumentParser(description="Naive Bayes evaluation")
    parser.add_argument('-i', '--input_file', help='Filename or path of the dataset', required=True)
    return parser

if __name__ == '__main__':
    main()
