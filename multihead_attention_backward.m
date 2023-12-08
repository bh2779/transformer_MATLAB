function [dQueries, dKeys, dValues] = multihead_attention_backward(dMultihead, X, wQ, wK, wV)
    dQueries = dMultihead * wQ';
    dKeys = dMultihead * wK';
    dValues = dMultihead * wV';
end