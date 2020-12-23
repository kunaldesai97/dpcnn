import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from scipy import stats

sns.set()

datasets = ['AG_News',
            'AmazonReviewPolarity',
            'AmazonReviewFull',
            'DBPedia',
            'SogouNews',
            'YelpReviewFull',
            'YelpReviewPolarity',
            ]

titles = ['AG News',
            'Amazon Review Polarity',
            'Amazon Review Full',
            'DBPedia',
            'Sogou News',
            'Yelp Review Full',
            'Yelp Review Polarity',
            ]


if __name__ == "__main__":
    
    os.chdir('Plotting')


    for dataset, title in zip(datasets, titles):
        paths = {
            'val_acc': "{}/run-.-tag-{}({})_batch_{}.csv".format(dataset,'Accuracy','Valid','acc'),
            'train_acc': "{}/run-.-tag-{}({})_batch_{}.csv".format(dataset,'Accuracy','Train','acc'),
            'val_los': "{}/run-.-tag-{}({})_batch_{}.csv".format(dataset,'Loss','Valid','loss'),
            'train_los': "{}/run-.-tag-{}({})_batch_{}.csv".format(dataset,'Loss','Train','loss'),
        }


        dfs = {}
        for p in paths.keys():
            dfs[p] = pd.read_csv(paths[p])
            
            # filter value data
            dfs[p]['Value'] = dfs[p]['Value'].ewm(alpha = .20).mean()


        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True)
        ax1.plot(dfs['val_acc']['Step'], dfs['val_acc']['Value'])
        ax1.set_title('Validiation')
        ax1.set(ylabel='Accuracy')
        ax2.plot(dfs['train_acc']['Step'], dfs['train_acc']['Value'])
        ax2.set_title('Training')
        fig.suptitle('{} Accuracy'.format(title))
        plt.savefig('Figures/{}_Acc.png'.format(dataset))


        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True)
        ax1.plot(dfs['val_los']['Step'], dfs['val_los']['Value'])
        ax1.set_title('Validiation')
        ax1.set(ylabel='Loss')
        ymax = max(dfs['val_los']['Value'])
        ax1.set_ylim([-ymax*.05,ymax*1.05])
        ax2.plot(dfs['train_los']['Step'], dfs['train_los']['Value'])
        ax2.set_title('Training')
        fig.suptitle('{} Loss'.format(title))
        plt.savefig('Figures/{}_Loss.png'.format(dataset))