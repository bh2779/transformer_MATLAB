function weights = weight_initialization(seq_len)
    n = 64*3*seq_len;

    weights.layer_1.attention.wQ = initializeHe([64 64], n);
    weights.layer_1.attention.wK = initializeHe([64 64], n);
    weights.layer_1.attention.wV = initializeHe([64 64], n);
    weights.layer_1.attention.wO = initializeHe([64 64], n);
    weights.layer_1.feedforward.w = initializeHe([64 64], n);
    weights.layer_1.feedforward.b = zeros(seq_len, 64);
    
    weights.output.feedforward.w = initializeHe([64 4], n);
    weights.output.feedforward.b = zeros(seq_len, 4);
end