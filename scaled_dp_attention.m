function attention = scaled_dp_attention(queries, keys, values)
    attention = softmax((queries * keys')/sqrt(size(queries,2))) * values;
end