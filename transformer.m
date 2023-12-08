function [z, outputs] = transformer(x,weights,outputs)
    z = cell(1,1);
    [temp, outputs.layer_1] = encoder(x, weights.layer_1, outputs.layer_1);
    z(1) = repelem({temp}, 1);
end