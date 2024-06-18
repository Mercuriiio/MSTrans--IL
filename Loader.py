import PIL
import os
import torch
from torch.utils import data
import pandas as pd
from util import *
import numpy as np
import random
from torchvision import transforms
from sklearn.model_selection import train_test_split

class PatchData(data.Dataset):
    def __init__(self, dataframe, split=None, transfer=None):
        self.dataframe = dataframe
        if split != None:
            index_split = self.dataframe[dataframe['Split'] == split].index
            self.dataframe = self.dataframe.loc[index_split, :]
        self.transfer = transfer
        self.length = len(self.dataframe)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        patch = self.dataframe.iloc[item, :]
        survival_label = patch[['days', 'event']].values.astype('float')
        # TCGA-GBMLGG
        if patch['Grade'] == 'G2':
            subtype_label = 0
        elif patch['Grade'] == 'G3':
            subtype_label = 1
        else:
            subtype_label = 2
        # TCGA-LUSC
        # if patch['Grade'] == 'N0':
        #     subtype_label = 0
        # elif patch['Grade'] == 'N1':
        #     subtype_label = 1
        # elif patch['Grade'] == 'N2':
        #     subtype_label = 2
        # else:
        #     subtype_label = 3
        # TCGA-BRCA
        # subtype_label = patch['Grade']

        slide_path_4096 = "./data/gbmlgg/4096/" + patch['img4096']
        slide_path_256 = "./data/gbmlgg/256/" + patch['img256']
        slide_path_16 = "./data/gbmlgg/16/" + patch['img16']
        slide_4096 = np.load(slide_path_4096)
        slide_256 = np.load(slide_path_256)
        slide_16 = np.load(slide_path_16)
        slide_4096 = np.transpose(slide_4096, (1, 2, 0))
        slide_256 = np.transpose(slide_256, (1, 2, 0))
        slide_16 = np.transpose(slide_16, (1, 2, 0))
        if self.transfer != None:
            slide_4096 = self.transfer(slide_4096)
            slide_256 = self.transfer(slide_256)
            slide_16 = self.transfer(slide_16)

        omics = patch.drop(["PatientID", "img", "Cluster", "Split", "days", "event", "Grade"]).values
        omics = torch.FloatTensor(omics.astype(float))

        # print('---------', slide_4096.shape, slide_256.shape, slide_16.shape)

        return slide_4096, slide_256, slide_16, survival_label, subtype_label, patch['PatientID']


    def split_cluster(file, split, cluster, transfer=None):
        df = pd.read_csv(file)
        index = df[df['Cluster'] == cluster].index

        return PatchData(df.loc[index, :], split=split, transfer=transfer)
