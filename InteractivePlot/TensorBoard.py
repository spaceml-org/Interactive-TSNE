from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch
import shutil
import torchvision.models as models
import numpy as np
import os
import cv2
from tensorboardX import SummaryWriter


class Projector(torch.nn.Module):

    '''
    This class extracts, reads, and writes data embeddings using a pretrained deep neural network. Meant to work with 
    Tensorboard's Embedding Viewer.
    When using with a 3 channel image input and a pretrained model from torchvision.models please use the 
    following pre-processing pipeline:
    
    transforms.Compose([transforms.Resize(imsize), 
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
    
    Args:
        model (nn.Module): A Pytorch model that returns an (B,1) embedding for a length B batched input
        LOG_PATH (str): The folder path where the necessary files will be written into
        EXPT_NAME (str): The name of the experiment to use as the log name
    '''

    def __init__(self, model, EXPT_NAME, LOG_PATH= '.'):
        
        super(Projector, self).__init__()
        self.model = model

        self.EXPT_NAME = EXPT_NAME
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.eval()
        if self.DEVICE == 'cuda':
            self.model.cuda()

        self.LOG_PATH = LOG_PATH
        self.SAVE_PATH = os.path.join(LOG_PATH, EXPT_NAME)
        self.IMGS_FOLDER = os.path.join(self.SAVE_PATH, "Images")
        self.EMBS_FOLDER = os.path.join(self.SAVE_PATH, "Embeddings")
        self.TB_FOLDER = os.path.join(self.SAVE_PATH, "TB")

        if not os.path.exists(self.SAVE_PATH):
            os.mkdir(self.SAVE_PATH)
            os.mkdir(self.IMGS_FOLDER)
            os.mkdir(self.EMBS_FOLDER)
            os.mkdir(self.TB_FOLDER)
        else: 
            shutil.rmtree(self.SAVE_PATH)
            os.mkdir(self.SAVE_PATH)
            os.mkdir(self.IMGS_FOLDER)
            os.mkdir(self.EMBS_FOLDER)
            os.mkdir(self.TB_FOLDER)

        self.writer = None
        
        
        
    
    def generate_embeddings(self, x):
        '''
        Generate embeddings for an input batched tensor
        
        Args:
            x (torch.Tensor) : A batched pytorch tensor
            
        Returns:
            (torch.Tensor): The output of self.model against x
        '''
        return(self.model(x))
    
    
    def write_embeddings(self, x, outsize=(32,32)):
        '''
        Generate embeddings for an input batched tensor and write inputs and 
        embeddings to self.IMGS_FOLDER and self.EMBS_FOLDER respectively. 
        
        Inputs and outputs will be stored in .npy format with randomly generated
        matching filenames for retrieval
        
        Args:
            x (torch.Tensor) : An input batched tensor that can be consumed by self.model
            outsize (tuple(int, int)) : A tuple indicating the size that input data arrays should be
            written out to
            
        Returns: 
            (bool) : True if writing was successful
        
        '''
        
        assert len(os.listdir(self.IMGS_FOLDER))==0, "Images folder must be empty"
        assert len(os.listdir(self.EMBS_FOLDER))==0, "Embeddings folder must be empty"
        if self.DEVICE == 'cuda':
            x= x.to('cuda')

        # Generate embeddings
        embs = self.generate_embeddings(x)
        
        # Detach from graph
        embs = embs.detach().cpu().numpy()
            
        # Start writing to output folders
        for i in range(len(embs)):
            key = str(np.random.random())[-7:]
            np.save(self.IMGS_FOLDER + r"/" + key + '.npy', tensor2np(x[i], outsize))
            np.save(self.EMBS_FOLDER + r"/" + key + '.npy', embs[i])
        return(True)
    
    
    def _create_writer(self, name):
        '''
        Create a TensorboardX writer object given an experiment name and assigns it to self.writer
        
        Args:
            name (str): Optional, an experiment name for the writer, defaults to self.name
        
        Returns:
            (bool): True if writer was created succesfully
        
        '''
        
        if self.EXPT_NAME is None:
            name = 'Experiment_' + str(np.random.random())
        else:
            name = self.EXPT_NAME
        
        dir_name = os.path.join(self.TB_FOLDER, 
                                name)
        
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        
        else:
            print("Warning: logfile already exists")
            print("logging directory: " + str(dir_name))
        
        logdir = dir_name
        self.writer = SummaryWriter(logdir=logdir)
        return(True)

    
    
    def create_tensorboard_log(self):
        
        '''
        Write all images and embeddings from imgs_folder and embs_folder into a tensorboard log
        '''
        
        if self.writer is None:
            self._create_writer(self.EXPT_NAME)
        
        
        ## Read in
        all_embeddings = [np.load(os.path.join(self.EMBS_FOLDER, p)) for p in os.listdir(self.EMBS_FOLDER) if p.endswith('.npy')]
        all_images = [np.load(os.path.join(self.IMGS_FOLDER, p)) for p in os.listdir(self.IMGS_FOLDER) if p.endswith('.npy')]
        all_images = [np.moveaxis(a, 2, 0) for a in all_images] # (HWC) -> (CHW)

        ## Stack into tensors
        all_embeddings = torch.Tensor(all_embeddings)
        all_images = torch.Tensor(all_images)

        print(all_embeddings.shape)
        print(all_images.shape)

        self.writer.add_embedding(all_embeddings, label_img = all_images)
        print('TensorBoard prepared. \n Open a new console and start TensorBoard with the logdir ', os.path.join(self.TB_FOLDER, self.EXPT_NAME))

        

def tensor2np(tensor, resize_to=None):
    '''
    Convert an image tensor to a numpy image array and resize
    
    Args:
        tensor (torch.Tensor): The input tensor that should be converted
        resize_to (tuple(int, int)): The desired output size of the array
        
    Returns:
        (np.ndarray): The input tensor converted to a channel last resized array
    '''
    
    out_array = tensor.detach().cpu().numpy()
    out_array = np.moveaxis(out_array, 0, 2) # (CHW) -> (HWC)
    
    if resize_to is not None:
        out_array = cv2.resize(out_array, dsize=resize_to, interpolation=cv2.INTER_CUBIC)
    
    return(out_array)