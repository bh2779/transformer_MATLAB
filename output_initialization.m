function outputs = output_initialization(seq_len)
    outputs.layer_1.mha = zeros(seq_len, 64);
    outputs.layer_1.attention = zeros(seq_len, 64);
    outputs.layer_1.n1 = zeros(seq_len, 64);
    outputs.layer_1.std1 = zeros(seq_len, 1);
    outputs.layer_1.feedforward = zeros(seq_len, 64);
    outputs.layer_1.n2 = zeros(seq_len, 64);
    outputs.layer_1.std2 = zeros(seq_len, 1);
end