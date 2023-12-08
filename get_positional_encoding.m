function positional_encoding = get_positional_encoding(seq_len, d, n)
    positional_encoding = zeros(seq_len, d);
    for k = 1:seq_len
        for i = 1:floor(d/n)
            denominator = n^(2*(i-1)/d);
            positional_encoding(k,2*(i-1)) = sin((k-1)/denominator);
            positional_encoding(k,2*i-1) = cos((k-1)/denominator);
        end
    end
end