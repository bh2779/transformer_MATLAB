function [dX, dWeights] = self_attention_backward(dA, X, weights, outputs)
    [dQ, dK, dV] = multihead_attention_backward(dA, X, weights.wQ, weights.wK, weights.wV);
    dX = dQ * weights.wQ' + dK * weights.wK' + dV * weights.wV';
    
    dWeights.dwQ = X' * dQ;
    dWeights.dwK = X' * dK;
    dWeights.dwV = X' * dV;
    dWeights.dwO = outputs.mha' * dA * weights.wO';
end