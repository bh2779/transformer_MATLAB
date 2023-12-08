function Z = feedforward(X,weights)
    Z = X * weights.w + weights.b;
    Z = custom_tanh(Z);
end