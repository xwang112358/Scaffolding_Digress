from dataset.get_datasets import get_dataset
import argparse
import os
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ogbg-molbace')

args = parser.parse_args()


labeled_dataset = get_dataset(args, './raw_data')
print(len(labeled_dataset))
# print(labeled_dataset[1200].y.item())

# print(labeled_dataset.scaff_list)

clustering = labeled_dataset.scaffold_clustering(cutoff=0.6)

indices = labeled_dataset.augmentation_sampling(N=100, plot=True, plot_path='./figures')
print(indices)

# print(clustering)


# task 
# check if the clustering is correct 
# plot a histogram to indicate class distribution in each cluster
