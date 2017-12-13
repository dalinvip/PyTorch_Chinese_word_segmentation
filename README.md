## Introduction
	pytorch implement  Chinese Word Segmentation and POS Tagging using seq2seq model

## Requirement
	* python 3
	* pytorch > 0.1(I used pytorch 0.2.0 and 0.3.0)
	* numpy

## Performance  Result
	update later

## Data
	CTB6.0

## How to run
	"python main-hyperparams.py"  or  "bash run.sh"


## How to use the folder or file

- the file of **`hyperparams.py`** contains all hyperparams that need to modify, based on yours nedds, select neural networks what you want and config the hyperparams.

- the file of **`main-hyperparams.py`** is the main function,run the command ("python main_hyperparams.py") to execute the demo.

- the folder of **`models`** contains encoder model and decoder model.

- the file of **`train_seq2seq_wordlstm_batch.py`** is the train function about batch decoder(fast speed)

- the file of **`train_seq2seq_wordlstm_nobatch.py`** is the train function about no use batch(low speed)

- the file of **`eval.py`** is the eval function about calculate F-score

- the folder of **`loaddata`** contains some file of load dataset

	'''  
	1、Dataloader.py  --- loading sorted data by sentence length  
	2、Dataloader_normal.py --- loading  data by order  
	3、Batch_Iterator.py --- create batch and iterator  
	4、Alphabet.py  --- cerate alphabet by data  
	5、Load_external_word_embedding.py  ---  load pretrained word embedding  
	'''

- the folder of **`word_embedding`** is the file of word embedding that have pretrained you want to use

- the folder of **`pos_test_data`** contains the dataset file, dataset is `CTB6.0`

- the file of **`Parameters.txt`** is being used to save all parameters values.


## How to use the Word Embedding in demo? 

- the word embedding file saved in the folder of **word_embedding**, but now is empty, because of it is to big,so if you want to use word embedding,you can to download word2vec or glove file, then saved in the folder of word_embedding,and make the option of xxx-Embedding to True and modifiy the value of xxx-Embedding_Path in the **hyperparams.py** file.


## Neural  Networks

	 seq2seq model

## How to config hyperparams in the file of hyperparams.py

- **learning_rate**: initial learning rate.

- **learning_rate_decay**: change the learning rate for optim.

- **epochs**:number of epochs for train

- **train-batch-size、dev-batch-size、test-batch-size**：batch size for train、dev、test

- **log_interval**：how many steps to wait before logging training status

- **test_interval**：how many steps to wait before testing

- **dev_interval**：how many steps to wait before dev

- **save_dir**：where to save the snapshot

- **train_path、dev-path、test-path**：datafile path

- **shuffle**:whether to shuffle the dataset when readed dataset 

- **dropout**:the probability for dropout

- **max_norm**:l2 constraint of parameters

- **clip-max-norm**:the values of prevent the explosion and Vanishing in Gradient

- **Adam**:select the optimizer of adam

- **SGD**：select the optimizer of SGD

- **Adadelta**:select the optimizer of Adadelta

- **optim-momentum-value**:the parameter in the optimizer

- **xxx-min-freq**:min freq to include during built the vocab

- **xxx-Embedding**: use word embedding

- **fix-Embedding**: use word embedding if to fix during trainging

- **embed-dim、embed-xxx-dim**:number of embedding dimension

- **xxx-Embedding-Path**:the path of word embedding file

- **num-layers**:the num of  layers with lstm

- **use-cuda**:  use cuda

- **num_threads**:set the value of threads when run the demo

- **init_weight**:whether to init weight

- **init-weight-value**:the value of init weight

- **weight-decay**:L2 weight_decay,default value is zero in optimizer

- **seed_num**:set the num of random seed

- **rm-model**:whether to delete the model after test acc so that to save space


## Reference 

	update later

