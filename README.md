# Master thesis - Feasibility of deep learning-based methods for the detection of simulated ECGs

Repository contains the code to train and test the models developed during the master thesis.  The thesis report can be accessed via the xAI chairs [webpage](https://www.uni-bamberg.de/xai/studium/abschlussarbeiten/).

## Structure:


    .
    ├── classification              # Files describing the models, how they where trained and tested
    │   ├── CNN-LSTM                # All files regarding the CNN-LSTM model
    │   └── CNN-Transformer         # All files regarding the CNN-Transformer model
    ├── generative_approach         # Contains all models, loss functions, training and demo scripts regarding the generative approach


## Reqired Packages:

* Defibrillator-Preprocessing ([https://github.com/Detecting-Simulated-Real-ECG-Signals/preprocessing](https://github.com/Detecting-Simulated-Real-ECG-Signals/preprocessing))
* LocalDataManagementTool ([https://github.com/Detecting-Simulated-Real-ECG-Signals/localdatamanagementtool](https://github.com/Detecting-Simulated-Real-ECG-Signals/DataManagement))
* jupyterlab
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
  -h, --help                                                    show this help message and exit
  -p PREPROCESSING, --preprocessing PREPROCESSING               If preprocessing is not defined, script uses default 'base' preprocessing.
  --device DEVICE                                               Device to run the model on. Options: cuda, cpu
  -lr LEARNING_RATE, --learning-rate LEARNING_RATE              Learning rate used to optimize the model.
  --d_model D_MODEL                                             d_model size of the transformer model.
  --stride STRIDE                                               Stride used by the CNN backbone.
  --kernel_size KERNEL_SIZE                                     Kernal size of the CNN backbone.
  --nhead NHEAD                                                 Number of multi headed attention used by the transformer.
  --num_layers NUM_LAYERS                                       Number of Transformer layers.
  --width_multiplier WIDTH_MULTIPLIER                           Channel multiplier used by the CNN backbone.
  --dropout DROPOUT                                             Dropout percentage.
  --FFN_dim_hidden_layers [FFN_DIM_HIDDEN_LAYERS ...]           Number of neurons used by the classification component used as hidden layer
  --validation-batches VALIDATION_BATCHES                       Amount of validation batches used to validate the models performance.
  --epochs EPOCHS                                               Number of training epochs.
  --num-workers NUM_WORKERS                                     Number of workers used to preprocess data.
  -b BATCH_SIZE, --batch-size BATCH_SIZE                        Batch size.
  --mlflow-tracking-uri MLFLOW_TRACKING_URI                     MLflow tracking uri.
  --mlflow-experiment MLFLOW_EXPERIMENT                         MLflow experiment name.                        
```


**Testing**
```console
python classification/CNN-Transformer/test_CNNTransformer.py
-h, --help                                                      show this help message and exit
  --device DEVICE                                               Device to run model on. Options: cuda, cpu
  -p [PREPROCESSING ...], --preprocessing [PREPROCESSING ...]   If preprocessing is not defined, script uses default 'base' preprocessing.
  -d DATABASE, --database DATABASE                              The used database must contain the preprocessing selected with '--preprocessing'. This argument requires a path to the preprocessing files.
  -r RUN_ID, --run-id RUN_ID                                    MLflow run id
  -o OUTPUT, --output OUTPUT                                    Output folder of the testing performance. Default: 'output/{run-id}'
  --num-workers NUM_WORKERS                                     Number of workers used to preprocess data.
  -b BATCH_SIZE, --batch-size BATCH_SIZE                        Batch size
  --mlflow-tracking-uri MLFLOW_TRACKING_URI                     MLflow tracking uri
  --mlflow-experiment MLFLOW_EXPERIMENT                         MLflow experiment name
```
you can state multiple preprocessings at one, this will return the classification results for each preprocessing. The output folder should be set, because if the path already exists, a error will occur.

### CNN-LSTM

**Training**

```console
python classification/CNN-LSTM/train_CNNLSTM.py
  -h, --help                                                    Show this help message and exit
  -p PREPROCESSING, --preprocessing PREPROCESSING               If preprocessing is not defined, script uses default 'base' preprocessing.
  --device DEVICE                                               Device to run the model on. Options: cuda, cpu
  -lr LEARNING_RATE, --learning-rate LEARNING_RATE              Learning rate used to optimize the model.
  --bidirectional BIDIRECTIONAL                                 Set the model to uni or bidirectional.
  --validation-batches VALIDATION_BATCHES                       Amount of validation batches used to validate the models performance.
  --epochs EPOCHS                                               Number of training epochs.
  --num-workers NUM_WORKERS                                     Number of workers used to preprocess data.
  -b BATCH_SIZE, --batch-size BATCH_SIZE                        Batch size.
  --mlflow-tracking-uri MLFLOW_TRACKING_URI                     MLflow tracking uri.
  --mlflow-experiment MLFLOW_EXPERIMENT                         MLflow experiment name.                              
```

**Testing**
```console
python classification/CNN-Transformer/test_CNNTransformer.py
  -h, --help                                                    Show this help message and exit
  --device DEVICE                                               Device to run the model on. Options: cuda, cpu
  -p [PREPROCESSING ...], --preprocessing [PREPROCESSING ...]   If preprocessing is not defined, script uses default 'base' preprocessing.
  -d DATABASE, --database DATABASE                              The used database must contain the preprocessing selected with '--preprocessing'.This argument requires a path to the preprocessing files.
  -r RUN_ID, --run-id RUN_ID                                    MLflow run id
  -o OUTPUT, --output OUTPUT                                    Output folder of the testing performance. Default: 'output/{run-id}'
  --num-workers NUM_WORKERS                                     Number of workers used to preprocess data.
  -b BATCH_SIZE, --batch-size BATCH_SIZE                        Batch size
  --mlflow-tracking-uri MLFLOW_TRACKING_URI                     MLflow tracking uri
  --mlflow-experiment MLFLOW_EXPERIMENT                         MLflow experiment name
```
you can state multiple preprocessings at one, this will return the classification results for each preprocessing. The output folder should be set, because if the path already exists, a error will occur.