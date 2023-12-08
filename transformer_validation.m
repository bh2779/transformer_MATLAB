function accuracy = transformer_validation(val_data, weights, outputs, seq_len)
    correct = 0;
    total = 0;
    for i = 1:size(val_data,1)
        total =  total + size(val_data{i},1);
        for j = 1:(size(val_data{i},1) - seq_len + 1)
            a = transformer(val_data{i}(j:j+seq_len-1,:), weights, outputs);
            z = a{1} * weights.output.feedforward.w + weights.output.feedforward.b;
            output = softmax(z')';
            [m, pred] = max(output, [], 2);
    
            if j == size(val_data{i},1) - seq_len + 1
                actual = val_data{i,2}(j:j+seq_len-1);
                correct = correct + sum(pred == actual', 1);
            else
                actual = val_data{i,2}(j);
                correct = correct + (pred(1) == actual);
            end
        end
    end
    accuracy = correct/total;
end