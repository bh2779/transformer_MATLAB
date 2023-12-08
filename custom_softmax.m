function output = custom_softmax(x)
    output = exp(x)/sum(exp(x));
end