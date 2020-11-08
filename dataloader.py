import data.ABIDEParser as Reader
import numpy as np
import torch
from utils.gcn_utils import preprocess_features
from sklearn.model_selection import StratifiedKFold


class dataloader():
    def __init__(self): 
        self.pd_dict = {}
        self.node_ftr_dim = 2000
        self.num_classes = 2 

    def load_data(self, connectivity='correlation', atlas='ho'):
        ''' load multimodal data from ABIDE
        return: imaging features (raw), labels, non-image data
        '''
        subject_IDs = Reader.get_ids()
        labels = Reader.get_subject_score(subject_IDs, score='DX_GROUP')
        num_nodes = len(subject_IDs)

        sites = Reader.get_subject_score(subject_IDs, score='SITE_ID')
        unique = np.unique(list(sites.values())).tolist()
        ages = Reader.get_subject_score(subject_IDs, score='AGE_AT_SCAN')
        genders = Reader.get_subject_score(subject_IDs, score='SEX') 

        y_onehot = np.zeros([num_nodes, self.num_classes])
        y = np.zeros([num_nodes])
        site = np.zeros([num_nodes], dtype=np.int)
        age = np.zeros([num_nodes], dtype=np.float32)
        gender = np.zeros([num_nodes], dtype=np.int)
        for i in range(num_nodes):
            y_onehot[i, int(labels[subject_IDs[i]])-1] = 1
            y[i] = int(labels[subject_IDs[i]])
            site[i] = unique.index(sites[subject_IDs[i]])
            age[i] = float(ages[subject_IDs[i]])
            gender[i] = genders[subject_IDs[i]]
        
        self.y = y -1  

        self.raw_features = Reader.get_networks(subject_IDs, kind=connectivity, atlas_name=atlas)

        phonetic_data = np.zeros([num_nodes, 3], dtype=np.float32)
        phonetic_data[:,0] = site 
        phonetic_data[:,1] = gender 
        phonetic_data[:,2] = age 

        self.pd_dict['SITE_ID'] = np.copy(phonetic_data[:,0])
        self.pd_dict['SEX'] = np.copy(phonetic_data[:,1])
        self.pd_dict['AGE_AT_SCAN'] = np.copy(phonetic_data[:,2]) 

        return self.raw_features, self.y, phonetic_data 

    def data_split(self, n_folds):
        # split data by k-fold CV
        skf = StratifiedKFold(n_splits=n_folds)
        cv_splits = list(skf.split(self.raw_features, self.y))
        return cv_splits 

    def get_node_features(self, train_ind):
        '''preprocess node features for ev-gcn
        '''
        node_ftr = Reader.feature_selection(self.raw_features, self.y, train_ind, self.node_ftr_dim)
        self.node_ftr = preprocess_features(node_ftr) 
        return self.node_ftr

    def get_PAE_inputs(self, nonimg):
        '''get PAE inputs for ev-gcn 
        '''
        # construct edge network inputs 
        n = self.node_ftr.shape[0] 
        num_edge = n*(1+n)//2 - n  
        pd_ftr_dim = nonimg.shape[1]
        edge_index = np.zeros([2, num_edge], dtype=np.int64) 
        edgenet_input = np.zeros([num_edge, 2*pd_ftr_dim], dtype=np.float32)  
        aff_score = np.zeros(num_edge, dtype=np.float32)   
        # static affinity score used to pre-prune edges 
        aff_adj = Reader.get_static_affinity_adj(self.node_ftr, self.pd_dict)  
        flatten_ind = 0 
        for i in range(n):
            for j in range(i+1, n):
                edge_index[:,flatten_ind] = [i,j]
                edgenet_input[flatten_ind]  = np.concatenate((nonimg[i], nonimg[j]))
                aff_score[flatten_ind] = aff_adj[i][j]  
                flatten_ind +=1

        assert flatten_ind == num_edge, "Error in computing edge input"
        
        keep_ind = np.where(aff_score > 1.1)[0]  
        edge_index = edge_index[:, keep_ind]
        edgenet_input = edgenet_input[keep_ind]

        return edge_index, edgenet_input

