function dz = layer_normalization_backward(dz, z, std)
    N = size(z, 2);
    dstd = -sum(dz .* z, 2) ./ (std.^2 + 1e-200);
    dmean = -sum(dz, 2) / N;
    dz = (dz + dstd .* 2 .* z / N + dmean) ./ (std + 1e-200);
end