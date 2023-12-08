function [z, std] = layer_normalization(z)
    mean = sum(z,2)/size(z,2);
    variance = 0;
    for i = 1:size(z,2)
        variance = variance + (z(:,i) - mean(:,1)).^2;
        z(:,i) = z(:,i) - mean(:,1);
    end
    std = sqrt(variance/size(z,2));
    z = z./(std + 1e-200);
end