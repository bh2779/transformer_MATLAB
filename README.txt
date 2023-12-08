Before running any of the functions/scripts, make sure the current folder is the folder enclosing the functions/scripts. To check the current folder, use the command 'cd'. To switch to another folder, use the command 'cd <path to folder>'.

To run just the training, run the transformer_training function. The input arguments are:
- training_data: embedded training data
- learning_rate
- epochs
- seq_len: the length of the sequence that the transformer looks at
The function returns are:
- weights: struct containing weights and biases for attention layers and feedforward layers
- outputs: struct containing outputs obtained at each layer

To see performance on the validation data, run the transformer_validation function. The input arguments are:
- val_data: embedded validation data
- weights: weights struct obtained from transformer_training
- outputs: outputs struct obtained from transformer_training
- seq_len: the length of the sequence that the transformer looks at
The function returns are:
- accuracy: the fraction of correctly predicted labels

To see performance on the test data, run the transformer_test function. The input arguments are:
- test_data: embedded test data
- weights: weights struct obtained from transformer_training
- outputs: outputs struct obtained from transformer_training
- seq_len: the length of the sequence that the transformer looks at
The function returns are:
- accuracy: the fraction of correctly predicted labels

To run the entire training process and get the accuracy of the predictions for the validation and test data, run the transformer_pipeline script. The values displayed at the end will be decimal values between 0 and 1.

The hyperparameters that I found to yield the best results were:
- learning_rate = 0.001
- epochs = 1
- seq_len = 2
These hyper parameters yielded an accuracy of 0.5809 on the validation data and 0.5875 on the training data