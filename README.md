# Flexible LSTM SequenceGenerator Wrapper

A simple wrapper to create a SequenceGenerator Machine learning using LSTM Networks in TfLearn.

It can be use to generate any text or number sequences just by tuning the parameters.

```bash
usage: lstm.py [-h] --dataset DATASET [--batch_size BATCH_SIZE]
               [--epochs EPOCHS] [--temperature TEMPERATURE]
               [--model_file MODEL_FILE]
               [--hidden_layer_size HIDDEN_LAYER_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     Path to the dataset file
  --batch_size BATCH_SIZE
                        How many string train on at a time
  --epochs EPOCHS       How many epochs to train
  --temperature TEMPERATURE
                        Temperature for generating the predictions
  --model_file MODEL_FILE
                        Path to save the model file, will be loaded if present
                        or created
  --hidden_layer_size HIDDEN_LAYER_SIZE
                        Number of hidden lstm layers
```
