function [dX, dWeights] = feedforward_backward(dZ, X, weights)
    dZ_tanh = dZ .* (1 - custom_tanh(X).^2);
    dX = dZ_tanh * weights.w';
    dWeights.dw = X' * dZ_tanh;
    dWeights.db = sum(dZ_tanh, 1);
end