*This project is adapted for Assignment 2, Course CZ4045: NLP at NTU*

**Team members**

+ KOH YIANG DHEE, MITCHELL
+ RYAN LEE RUIXIANG
+ LI PINGRUI
+ WANG BINLI
+ TANG YUTING

# Dependency

```
numpy~=1.21.3
gsdmm~=0.1
setuptools~=58.2.0
nltk~=3.6.5
spacy~=3.1.3
seaborn~=0.11.2
matplotlib~=3.4.3
scikit-learn~=1.0
tqdm~=4.62.3
wordcloud~=1.8.1
torch 
```

To install the dependencies, run

```
pip install -r requirements.txt
```


# End-to-end-Sequence-Labeling-via-Bi-directional-LSTM-CNNs-CRF-Tutorial

This is a PyTorch tutorial for the ACL'16 paper 
[**End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF**](http://www.aclweb.org/anthology/P16-1101)

This repository includes

* [**IPython Notebook of the tutorial**](https://github.com/jayavardhanr/End-to-end-Sequence-Labeling-via-Bi-directional-LSTM-CNNs-CRF-Tutorial/blob/master/Named_Entity_Recognition-LSTM-CNN-CRF-Tutorial.ipynb)
* Data folder
* Setup Instructions file
* Pretrained models directory (The notebook will automatically download pre-trained models into this directory, as required)

### Authors

[**Anirudh Ganesh**](https://github.com/TheAnig)

[**Peddamail Jayavardhan Reddy**](https://github.com/jayavardhanr)


### Installation
The best way to install pytorch is via the [**pytorch webpage**](http://pytorch.org/)

### Setup

#### Creating new Conda environment
`conda create -n pytorch python=3.5`

#### Activate the condo environment
`source activate pytorch`

#### Setting up notebooks with specific python version (python 3.5)
```
conda install notebook ipykernel
ipython kernel install --user
```

#### PyTorch Installation command:
`conda install pytorch torchvision -c pytorch`

#### NumPy installation
`conda install -c anaconda numpy`

#### Download GloVe vectors and extract glove.6B.100d.txt into "./data/" folder

`wget http://nlp.stanford.edu/data/glove.6B.zip`

#### Data Files


You can download the data files from within this repo [**over here**](https://github.com/TheAnig/NER-LSTM-CNN-Pytorch/tree/master/data)

#### Run ipynb notebook without retraining model

Under Training Step cell, In [31], comment the first line of code : parameters['reload']=False

i.e #parameters['reload']=False

This will reload the pretrained models for the respective implementations from the models file. Model testing code left out from the original notebook can be added to try out the trained models.

Model Testing code can be found : https://github.com/jayavardhanr/End-to-end-Sequence-Labeling-via-Bi-directional-LSTM-CNNs-CRF-Tutorial/blob/master/Named_Entity_Recognition-LSTM-CNN-CRF-Tutorial.ipynb