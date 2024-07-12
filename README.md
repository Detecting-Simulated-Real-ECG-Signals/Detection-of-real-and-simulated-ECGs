# Master thesis - Feasibility of deep learning-based methods for the detection of simulated ECGs

Repository contains the code to train and test the models developed during the master thesis.  The thesis report can be accessed via the xAI chairs [webpage](https://www.uni-bamberg.de/xai/studium/abschlussarbeiten/).

## Structure:

* classification -> CNN-Transformer and CNN-LSTM files describing the models. Also contains training and testing scripts
* generative_approach -> Contains all models, loss functions, training and demo scripts regarding the generative approach


## Reqired Packages:

* Defibrillator-Preprocessing ([https://github.com/Detecting-Simulated-Real-ECG-Signals/preprocessing](https://github.com/Detecting-Simulated-Real-ECG-Signals/preprocessing))
* LocalDataManagementTool ([https://github.com/Detecting-Simulated-Real-ECG-Signals/localdatamanagementtool](https://github.com/Detecting-Simulated-Real-ECG-Signals/DataManagement))
* jupyter
* scikit
* sklearn
* torch (https://pytorch.org/get-started/locally/)
* torchmetrics
* tqdm



## Important scripts

### CNN-Transformer

**Training**

```console
python classification/CNN-Transformer/train_CNNTransformer.py
  -p <PREPROCESSING>, --preprocessing <PREPROCESSING>         -> set preprocessing                                
  --device <DEVICE>                                           -> set device (Options: cuda, cpu)
  -lr <LEARNING_RATE>, --learning-rate <LEARNING_RATE>        -> set lr (default: 10^-4)                                      
  --d_model D_MODEL                                           -> channel into the transformer model (default: 64) 
  --stride STRIDE                                             -> CNN stride  (default: 2)
  --kernel_size <KERNEL_SIZE>                                 -> CNN kernal-size (default: 3)             
  --nhead <NHEAD>                                             -> Multi-Headed Attention number of heads (default: 8)
  --num_layers <NUM_LAYERS>                                   -> CNN-Layers (default: 6)         
  --width_multiplier <WIDTH_MULTIPLIER>                       -> SE width multiplier (default: 6)                     
  --dropout <DROPOUT>                                         -> Dropout percentage     
  --FFN_dim_hidden_layers [<FFN_DIM_HIDDEN_LAYERS> ...]       -> Number of neurons used in the middle layer of the classifier                                      
  --epochs <EPOCHS>                                           -> Training Mini-Epochs 
  -b BATCH_SIZE, --batch-size <BATCH_SIZE>                    -> Batch-size                         
  --validation-batches <VALIDATION_BATCHES>                   -> Number of batches used to validate trainings progress                         
  --num-workers <NUM_WORKERS>                                   
  --mlflow-tracking-uri <MLFLOW_TRACKING_URI>                                   
  --mlflow-experiment <MLFLOW_EXPERIMENT>                                   
```


**Testing**
```console
python classification/CNN-Transformer/test_CNNTransformer.py
  -r <run-id:str>                                             -> mlflow run id
  -d <database-if-not-default:str>                            -> path of database from the bib localdataset  
  -p <data-preprocessing-like-windows:str>                    -> data preprocessing available in the selected database
  -b <batch-size:int>                                         -> batch size
  --mlflow-tracking-uri <MLFLOW_TRACKING_URI:str>             -> mlflow tracking uri
  -o <output-folder:str>      
```
you can state multiple preprocessings at one, this will return the classification results for each preprocessing. The output folder should be set, because if the path already exists, a error will occur.

### CNN-LSTM

**Training**

```console
python classification/CNN-LSTM/train_CNNLSTM.py
  -p <PREPROCESSING>, --preprocessing <PREPROCESSING>         -> set preprocessing                                
  --device <DEVICE>                                           -> set device (Options: cuda, cpu)
  -lr <LEARNING_RATE>, --learning-rate <LEARNING_RATE>        -> set lr (default: 10^-4)
  --bidirectional <BIDIRECTIONAL:bool>                        -> bidirectional LSTM (default: True)  
  --epochs <EPOCHS>                                           -> Training Mini-Epochs 
  -b BATCH_SIZE, --batch-size <BATCH_SIZE>                    -> Batch-size                         
  --validation-batches <VALIDATION_BATCHES>                   -> Number of batches used to validate trainings progress                         
  --num-workers <NUM_WORKERS>                                   
  --mlflow-tracking-uri <MLFLOW_TRACKING_URI>                                   
  --mlflow-experiment <MLFLOW_EXPERIMENT>                                   
```

**Testing**
```console
python classification/CNN-Transformer/test_CNNTransformer.py
  -r <run-id:str>                                             -> mlflow run id
  -d <database-if-not-default:str>                            -> path of database from the bib localdataset  
  -p <data-preprocessing-like-windows:str>                    -> data preprocessing available in the selected database
  -b <batch-size:int>                                         -> batch size
  --mlflow-tracking-uri <MLFLOW_TRACKING_URI:str>             -> mlflow tracking uri
  -o <output-folder:str>                                      -> output folder of the classification results
```
you can state multiple preprocessings at one, this will return the classification results for each preprocessing. The output folder should be set, because if the path already exists, a error will occur.
