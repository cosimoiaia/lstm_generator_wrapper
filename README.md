# Flexible LSTM SequenceGenerator Wrapper

A simple wrapper to flexibly create a SequenceGenerator Machine learning using LSTM Networks in TfLearn.

It can be use to generate any text or number sequences just by tuning the parameters.

i.e. City Name Generation or Shakespeare Scripts Generation from http://tflearn.org/examples/#natural-language-processing

The trained model will be automatically saved each run.
Batch size, number of epochs and temperature for generations can be passed as commandline parameters, 
as the number of hidden lstm layers.

```bash
usage: lstm.py [-h] --dataset DATASET [--batch_size BATCH_SIZE]
               [--epochs EPOCHS] [--temperature TEMPERATURE]
               [--model_file MODEL_FILE]
               [--hidden_layer_size HIDDEN_LAYER_SIZE]
               [--max_sequence_lenght MAX_SEQUENCE_LENGHT]

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
  --max_sequence_lenght MAX_SEQUENCE_LENGHT
                        Max lenght of sequence sample. if 0 is set to the
                        maximum sequence lenght from dataset (Warning:
                        requires serious amount of Memory)
```
