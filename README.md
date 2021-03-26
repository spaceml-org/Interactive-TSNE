# Interactive Viewer

Interactive Viewer is a tool that provides a way to visually view a PyTorch model's feature representation for better embedding space interpretability.

# Features

1. An Interactive plot to view the nearest neighbors of every data point in an embeddding space. Currently supports TSNE/UMAP. 

2. Visualize data in 3D using the TensorBoard Embedding Projector.

# Installation

``` 
pip install git+https://github.com/spaceml-org/Interactive-TSNE.git
```

# Interactive Plot 



![alt text](https://s4.gifyu.com/images/2021-03-24-03-33-49-2.gif "Interactive Plot")


## Usage

```
from InteractivePlot import PrepareData
from InteractivePlot import InteractiveAllclose
%matplotlib notebook

model = 'uc_merced.pt'
data = 'UCMerced_LandUse/Images'

data = PrepareData(model = model, DATA_PATH= data, num_clusters= 21, output_size= 21)

p = InteractiveAllclose(tcl = data.cl, 
                        tsne_obj = data.tsne_obj, 
                        objects = data.objects, 
                        spd =data.spd)
```

num_clusters : the number of clusters to be mapped to a color scheme. For color coding purposes only.

output_size : the number of output dimensions of your model.

**NOTE** : This currently works only on Jupyter notebook instances that support either the __widget__ or the __notebook__ matplotlib backends. Does not currently support Colab. 


# TensorBoard Projector


<img src="TensorBoard.gif?raw=true" width="2000px">

## Usage 

```
#Initialize model and data

model = torchvision.model.resnet18(pretrained= True) #Load model
model.cuda()
model.eval()

tfs = transforms.Compose([transforms.Resize((128, 128)), 
                          transforms.ToTensor(),
                          transforms.Normalize(mean = [0.485], std = [0.229])])

dataset = FashionMNIST(root = r'./FMINST', download = True, transform= tfs)
data_loader = torch.utils.data.DataLoader(dataset, batch_size = 256, shuffle= True)
batch_imgs, batch_imgs = next(iter(data_loader))



#Start the projector
from InteractivePlot import Projector

vis = Projector(model = model, EXPT_NAME = 'projector_test', LOG_PATH = '.')
vis.write_embeddings(batch_imgs)
vis.create_tensorboard_log()

```

This will output a log directory where the TensorBoard files are written, and you can directly launch TensorBoard from that directory. 

```
tensorboard --logdir=output path
```



## Coming Updates
- Streamlit hosting
- TensorFlow models
- More Dimensionality Reduction methods
