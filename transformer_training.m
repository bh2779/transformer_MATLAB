function [weights, outputs] = transformer_training(training_data, learning_rate, epochs, seq_len)
    weights = weight_initialization(seq_len);
    outputs = output_initialization(seq_len);
    
    for epoch = 1:epochs
        for i = 1:size(training_data,1)
            for j = 1:(size(training_data{i},1) - seq_len + 1)
                [a, outputs] = transformer(training_data{i}(j:j+seq_len-1,:), weights, outputs);
                z = a{1} * weights.output.feedforward.w + weights.output.feedforward.b;
                pred = softmax(z')';
                actual = zeros(seq_len, 4);
                for k = 1:seq_len
                    tags = training_data{i,2}(j:j+seq_len-1);
                    actual(k,tags(k)) = 1;
                end
                
                dX = pred - actual;
                d.output.w = a{1}' * dX;
                d.output.b = dX;
    
                dX = dX * weights.output.feedforward.w';
    
                dX = layer_normalization_backward(dX, ...
                                                  outputs.layer_1.feedforward, ...
                                                  outputs.layer_1.std2);
                [dX, d.layer_1.ff] = feedforward_backward(dX, ...
                                                          outputs.layer_1.n1, ...
                                                          weights.layer_1.feedforward);
                dX = layer_normalization_backward(dX, ...
                                                  outputs.layer_1.attention, ...
                                                  outputs.layer_1.std1);
                [dX, d.layer_1.att] = self_attention_backward(dX, ...
                                                              training_data{i}(j:j+seq_len-1,:), ...
                                                              weights.layer_1.attention, ...,
                                                              outputs.layer_1);
    
                weights.output.feedforward.w = weights.output.feedforward.w - learning_rate * d.output.w;
                weights.output.feedforward.b = weights.output.feedforward.b - learning_rate * d.output.b;

                weights.layer_1.feedforward.w = weights.layer_1.feedforward.w - learning_rate * d.layer_1.ff.dw;
                weights.layer_1.feedforward.b = weights.layer_1.feedforward.b - learning_rate * d.layer_1.ff.db;
                weights.layer_1.attention.wO = weights.layer_1.attention.wO - learning_rate * d.layer_1.att.dwO;
                weights.layer_1.attention.wQ = weights.layer_1.attention.wQ - learning_rate * d.layer_1.att.dwQ;
                weights.layer_1.attention.wK = weights.layer_1.attention.wK - learning_rate * d.layer_1.att.dwK;
                weights.layer_1.attention.wV = weights.layer_1.attention.wV - learning_rate * d.layer_1.att.dwV;
            end
        end
    end
end
