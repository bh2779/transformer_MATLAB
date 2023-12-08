function [z, outputs] = encoder(z,weights, outputs)
    a = self_attention(z, weights.attention, outputs);
    outputs.attention = a;
    [z, std1] = layer_normalization(a + z);
    outputs.n1 = z;
    outputs.std1 = std1;
    a = feedforward(z, weights.feedforward);
    outputs.feedforward = a;
    [z, std2] = layer_normalization(a + z);
    outputs.n2 = z;
    outputs.std2 = std2;
end