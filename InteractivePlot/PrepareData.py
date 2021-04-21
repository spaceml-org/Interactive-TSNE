import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
import os
from tqdm.notebook import tqdm
import numpy as np
from PIL import Image
import pickle
from tqdm import tqdm
import random
import time
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import glob

from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import pdist 
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage

from matplotlib import pyplot as plt
from matplotlib import cm
from PIL import Image
from io import BytesIO
import traceback, functools
import pandas as pd 
import umap

class PrepareData:
    '''
    Prepares Data for passing to the Interactive TSNE. Specifically linkage variables used to represent relationship between clusters.
    '''

    def __init__(self, model, DATA_PATH, output_size, num_clusters, method = 'tsne', perplexity= 30, n_jobs = 4, n_neighbors= 5):
        self.model = model
        self.data_path = DATA_PATH
        self.embeddings = self.get_matrix(MODEL_PATH = model, DATA_PATH = DATA_PATH, output_size = output_size)
        self.ims = self.get_images(DATA_PATH)
        self.tsne_obj, self.spd, self.cl, self.objects = self.variable_gen(self.embeddings, num_clusters, self.ims, method, perplexity, n_jobs, n_neighbors)

    def get_matrix(self, MODEL_PATH, DATA_PATH, output_size):

        '''      
        Generate Embeddings for a given folder of images and a stored model.

        Args:
            MODEL_PATH (str) : Path to PyTorch Model
            DATA_PATH (str) : Path to ImageFolder Dataset
            output_size(int) : out_features value of the model
        
        Returns:
            (torch.Tensor) : Embeddings of the given model on the specified dataset.
        '''

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            print('Using CUDA')
            model = torch.load(MODEL_PATH)
        else: 
            print('Using CPU')
            model = torch.load(MODEL_PATH, map_location = torch.cpu())
        t = transforms.Compose(
        [transforms.Resize((224, 224)),
            transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,), (0.5,))])

        dataset = torchvision.datasets.ImageFolder(DATA_PATH, transform = t)
        model.eval()
        if device == 'cuda':
            model.cuda()
        with torch.no_grad():
            if device == 'cuda':
                data_matrix = torch.empty(size = (0, output_size)).cuda()
            else: 
                data_matrix = torch.empty(size = (0, output_size))
            bs = 64
            if len(dataset) < bs:
                bs = 1
            loader = torch.utils.data.DataLoader(dataset, batch_size = bs, shuffle = False)
            for batch in tqdm(loader):
                if device == 'cuda':
                    x = batch[0].cuda()
                else:
                    x = batch[0]
                embeddings = model(x)[0]
                data_matrix = torch.vstack((data_matrix, embeddings))
        return data_matrix.cpu().detach().numpy() 

    def get_images(self, DATA_PATH):

        '''
        Get list of images in a folder.

        Args:
            DATA_PATH (str) : Path to ImageFolder Dataset
        
        Returns:
            (list) : Order of images in a folder
        '''

        ims = []
        for folder in os.listdir(DATA_PATH):
            for im in os.listdir(f'{DATA_PATH}/{folder}'):
                ims.append(f'{DATA_PATH}/{folder}/{im}')
        return ims
    
    def fit_tsne(self, feature_list, perplexity= 30, n_jobs= 4):

        '''
        Fits TSNE for the input embeddings

        Args: 

            feature_list: ssl embeddings
            perplexity : hyperparameter that determines how many images are close to each other in a cluster
            n_jobs : number of jobs to be run concurrently.
        
        Returns: 
            (numpy.ndarray) : TSNE result vector
        '''
        
        n_components = 2
        verbose = 1
        perplexity = perplexity
        n_iter = 1000
        metric = 'euclidean'
        n_jobs= n_jobs

        time_start = time.time()
        tsne_results = TSNE(n_components=n_components,
                            verbose=verbose,
                            perplexity=perplexity,
                            n_iter=n_iter,
                            n_jobs= n_jobs,
                            random_state=42,
                            metric=metric).fit_transform(feature_list)

        print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))
        return tsne_results

    def fit_umap(self, feature_list, n_neighbors = 5, n_jobs = 4):
        time_start = time.time()
        fit = umap.UMAP(
            n_neighbors=n_neighbors,
            random_state=42,
            n_components=2,
            verbose = 1,
            n_jobs = n_jobs,
            metric='euclidean')
        
        u = fit.fit_transform(feature_list)
        print('UMAP done! Time elapsed: {} seconds'.format(time.time() - time_start))
        return u

    def calculate_similarity(self, embeddings, nclusters, method, perplexity, n_jobs, n_neighbors):

        '''
        Calculates necessary linkage variables for visualization.

        Args: 
            embeddings: Model features
            nclusters : Number of clusters present in our flat cluster. FOR COLOR CODING PURPOSE ONLY.
        
        Returns: 
           tsne_obj (numpy.ndarray) : t-SNE coordinate mapping
           spd (numpy.ndarray) : a square array [nobj,nobj] of distances
           cl (numpy.ndarray) : list of clusters
        '''

        z = linkage(embeddings, method = 'centroid')
        pdt = pdist(embeddings, metric= 'cosine')
        if method == 'umap':
            tsne_obj = self.fit_umap(embeddings, n_neighbors, n_jobs)
        elif method == 'tsne':
            tsne_obj = self.fit_tsne(embeddings, perplexity, n_jobs)
        spd = squareform(pdt)
        cl = fcluster(z.astype(float), nclusters, criterion= 'maxclust')
        print("Linkage variables created.")

        return tsne_obj, spd, cl
    
    def object_creation(self, ims):
        '''
        Creates a DataFrame containing filename label mapping

        Args: 
            ims (list) : Order of images in a folder

        Retuns: 
            objects (pandas.DataFrame)  
        '''
        df_lis = []
        for img in ims:
            df_lis.append((img.split('/')[-2], img))

        objects = pd.DataFrame(df_lis, columns = ['name', 'filename'])
        print("Filename mapping done.")
        return objects

    def variable_gen(self, embeddings, num_clusters, ims, method, perplexity, n_jobs, n_neighbors):
        '''
        Helper function to return variables
        '''

        tsne_obj, spd, cl = self.calculate_similarity(embeddings, num_clusters,method, perplexity, n_jobs, n_neighbors)
        objects = self.object_creation(ims)
        return tsne_obj, spd, cl, objects




