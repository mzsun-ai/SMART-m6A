import pickle
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
from Functions_seq import *
import matplotlib.pyplot as plt

tokens='ACGU().BEHIMSXN'
class RNADataset(Dataset):
    def __init__(self,seqs,labels,ids,ew,bpp_path,transform=None,training=True,pad=False,k=5):
        self.transform=transform
        self.seqs=seqs
        self.data=[]
        self.labels=labels.astype('float32')
        self.bpp_path=bpp_path
        self.ids=ids
        self.training=training
        self.bpps=[]
        dm=get_distance_mask(len(seqs[-1]))
        self.dms=np.asarray([dm for i in range(1)])
        self.lengths=[]
        for i,id in tqdm(enumerate(self.ids)):
            bpps=np.load(os.path.join(self.bpp_path,'bpps',str(id)+'_bpp.npy'))
            if pad:
                bpps=np.pad(bpps,([0,0],[0,130-bpps.shape[1]],[0,130-bpps.shape[2]]), constant_values=0)
            with open(os.path.join(self.bpp_path,'dotbrackets',str(id)+'_struct.p'),'rb') as f:
                structures=pickle.load(f)
            with open(os.path.join(self.bpp_path,'loops',str(id)+'_loop.p'),'rb') as f:
                loops=pickle.load(f)
            if isinstance(loops, list):
                loops = loops[0]  
            input_loop = np.asarray([tokens.index(s) for s in loops])

            seq=self.seqs[i]
            self.lengths.append(len(seq))
            input=[]
    
            input_seq = np.asarray([tokens.index(s) for s in seq])

            if isinstance(structures, list):
                structures = structures[0] 
            input_structure = np.asarray([tokens.index(s) for s in structures])
            input_loop = np.asarray([tokens.index(s) for s in loops])
            input=np.stack([input_seq,input_structure,input_loop],-1)
            input = np.expand_dims(input, axis=0).astype('int')
   
            if pad:
                input=np.pad(input,([0,0],[0,130-input.shape[1]],[0,0]), constant_values=14)
            self.data.append(input)
            self.bpps.append(np.clip(bpps,0,1).astype('float32'))

        self.data=np.asarray(self.data)
        self.lengths=np.asarray(self.lengths)

        self.ew=ew
        self.k=k

        self.src_masks=[self.generate_src_mask(self.lengths[i],self.data.shape[-2],self.k) for i in range(len(self.data))]
        self.pad=pad


    def generate_src_mask(self,L1,L2,k):
        mask=np.ones((k,L2),dtype='int8')
        for i in range(k):
            mask[i,L1+i+1-k:]=0
        return mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
 
        if self.training:
            bpps=self.bpps[idx]
            bpps = bpps[np.newaxis, :, :]
            bpps = np.concatenate([bpps, self.dms[0]], axis=0)
            bpps=bpps.astype('float32')
            data_input = self.data[idx][0]
            sample = {'data': data_input, 'labels': self.labels[idx], 'bpp': bpps,
            'ew': self.ew[idx],'id':self.ids[idx],'src_mask':self.src_masks[idx]}
        else:
            bpps=self.bpps[idx]
            bpps = bpps[np.newaxis, :, :] 
            bpps = np.concatenate([bpps, self.dms[0]], axis=0) 
            bpps=bpps.astype('float32')
            data_input = self.data[idx][0] 
            sample = {'data': data_input, 'labels': self.labels[idx], 'bpp': bpps,
           'ew': self.ew[idx],'id':self.ids[idx],'src_mask':self.src_masks[idx]}

        return sample
