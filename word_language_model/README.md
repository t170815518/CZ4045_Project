*This project is for Assignment 2, Course CZ4045: NLP at NTU*

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


# Word-level language modeling

This example trains a multi-layer RNN (Elman, GRU, or LSTM, FNN) on a language modeling task.
By default, the training script uses the Wikitext-2 dataset, provided.
The trained model can then be used by the generate script to generate new text.

```bash 
python main.py --cuda --epochs 6           # Train a LSTM on Wikitext-2 with CUDA
python main.py --cuda --epochs 6 --tied    # Train a tied LSTM on Wikitext-2 with CUDA
python main.py --cuda  --batch_size 100 --model FNN --lr 0.0001 
    --bptt 8 --nhid 100 --emsize 100 --optimizer Adam --tie  # Train FNN model 

python generate.py --cuda  --temperature 1  # generate text from pre-trained FNN model 
```

The model uses the `nn.RNN` module (and its sister modules `nn.GRU` and `nn.LSTM`)
which will automatically use the cuDNN backend if run on CUDA with cuDNN installed.

During training, if a keyboard interrupt (Ctrl-C) is received,
training is stopped and the current model is evaluated against the test dataset.

The `main.py` script accepts the following arguments:

```bash
optional arguments:
  -h, --help            show this help message and exit
  --data DATA           location of the data corpus
  --model MODEL         type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU,
                        Transformer)
  --emsize EMSIZE       size of word embeddings
  --nhid NHID           number of hidden units per layer
  --nlayers NLAYERS     number of layers
  --lr LR               initial learning rate
  --clip CLIP           gradient clipping
  --epochs EPOCHS       upper epoch limit
  --batch_size N        batch size
  --bptt BPTT           sequence length
  --dropout DROPOUT     dropout applied to layers (0 = no dropout)
  --tied                tie the word embedding and softmax weights
  --seed SEED           random seed
  --cuda                use CUDA
  --log-interval N      report interval
  --save SAVE           path to save the final model
  --onnx-export ONNX_EXPORT
                        path to export the final model in onnx format
  --nhead NHEAD         the number of heads in the encoder/decoder of the
                        transformer model
```

With these arguments, a variety of models can be tested.
As an example, the following arguments produce slower but better models:

```bash
python main.py --cuda --emsize 650 --nhid 650 --dropout 0.5 --epochs 40           
python main.py --cuda --emsize 650 --nhid 650 --dropout 0.5 --epochs 40 --tied    
python main.py --cuda --emsize 1500 --nhid 1500 --dropout 0.65 --epochs 40        
python main.py --cuda --emsize 1500 --nhid 1500 --dropout 0.65 --epochs 40 --tied 
```

Example Output: 
+ the arguments are print before training starts. 
+ the training information is echoed after each interval 
+ the valid dataset is tested after each epoch 
+ the test dataset is tested after the training 

```
main.py --cuda --batch_size 100 --model FNN --lr 0.0001 --bptt 8 --nhid 100 --emsize 30 --optimizer Adam
Namespace(batch_size=100, bptt=8, clip=0.25, connect_feature2output=False, cuda=True, data='./data/wikitext-2', dropout=0.2, dry_run=False, emsize=30, epochs=20, log_interval=200, lr=A, model='FNN', n_gram=8, nhead=2, nhid=100, nlayers=2, onnx_export='', optimizer='Adam', save='best_model.pt', seed=1111, tied=False)
| epoch   1 |   200/ 2610 batches | lr 0.00 | ms/batch  7.84 | loss 10.40 | ppl 32745.79
| epoch   1 |   400/ 2610 batches | lr 0.00 | ms/batch  5.22 | loss 10.04 | ppl 22951.71
| epoch   1 |   600/ 2610 batches | lr 0.00 | ms/batch  5.18 | loss  9.32 | ppl 11106.31
| epoch   1 |   800/ 2610 batches | lr 0.00 | ms/batch  5.19 | loss  8.21 | ppl  3672.89
| epoch   1 |  1000/ 2610 batches | lr 0.00 | ms/batch  5.14 | loss  7.54 | ppl  1881.11
| epoch   1 |  1200/ 2610 batches | lr 0.00 | ms/batch  5.05 | loss  7.44 | ppl  1705.17
| epoch   1 |  1400/ 2610 batches | lr 0.00 | ms/batch  5.16 | loss  7.40 | ppl  1641.80
| epoch   1 |  1600/ 2610 batches | lr 0.00 | ms/batch  5.10 | loss  7.32 | ppl  1505.64
| epoch   1 |  1800/ 2610 batches | lr 0.00 | ms/batch  5.08 | loss  7.31 | ppl  1490.09
| epoch   1 |  2000/ 2610 batches | lr 0.00 | ms/batch  5.06 | loss  7.25 | ppl  1413.16
| epoch   1 |  2200/ 2610 batches | lr 0.00 | ms/batch  5.09 | loss  7.26 | ppl  1427.37
| epoch   1 |  2400/ 2610 batches | lr 0.00 | ms/batch  5.24 | loss  7.22 | ppl  1367.77
| epoch   1 |  2600/ 2610 batches | lr 0.00 | ms/batch  5.30 | loss  7.19 | ppl  1329.12
-----------------------------------------------------------------------------------------
| end of epoch   1 | time: 15.23s | valid loss  6.07 | valid ppl   432.54
```
