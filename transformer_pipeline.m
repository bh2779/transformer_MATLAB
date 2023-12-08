embeddings = get_embeddings('wv.csv');
training_data = get_data('train_data.csv', embeddings);
val_data = get_data('valid_data.csv', embeddings);
test_data = get_data('test_data.csv', embeddings);

learning_rate = 0.001;
epochs = 1;
seq_len = 2;

[weights, outputs] = transformer_training(training_data, learning_rate, epochs, seq_len);
val_accuracy = transformer_validation(val_data, weights, outputs, seq_len);
disp('Accuracy on validation data:');
disp(val_accuracy);
test_accuracy = transformer_test(test_data, weights, outputs, seq_len);
disp('Accuracy on test data:');
disp(test_accuracy);