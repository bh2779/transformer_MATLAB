function output = custom_tanh(x)
    output = (power(exp(1),x) - power(exp(1),-x))./(power(exp(1),x) + power(exp(1),-x));
end