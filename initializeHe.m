function weights = initializeHe(s, n)
    weights = randn(s) * sqrt(2/n);
end