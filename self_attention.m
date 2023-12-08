function [A, outputs] = self_attention(X, weights, outputs)
    Q = X * weights.wQ;
    K = X * weights.wK;
    V = X * weights.wV;

    Q = permute(reshape(Q, size(X,1), 2, 32), [2 1 3]);
    K = permute(reshape(K, size(X,1), 2, 32), [2 1 3]);
    V = permute(reshape(V, size(X,1), 2, 32), [2 1 3]);

    A = multihead_attention(Q, K, V) * weights.wO;
    outputs.mha = A;
end