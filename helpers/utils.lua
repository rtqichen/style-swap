-- Prepares an RGB image in [0,1] for VGG
function getPreprocessConv()
    local mean_pixel = torch.Tensor({103.939, 116.779, 123.68})
    local conv = nn.SpatialConvolution(3,3, 1,1)
    conv.weight:zero()
    conv.weight[{1,3}] = 255
    conv.weight[{2,2}] = 255
    conv.weight[{3,1}] = 255
    conv.bias = -mean_pixel
    conv.gradBias = nil
    conv.gradWeight = nil
    conv.parameters = function() --[[nop]] end
    conv.accGradParameters = function() --[[nop]] end
    return conv
end

function getTVDenoisingConv(strength)
    local conv = nn.SpatialConvolution(3,3, 3,3, 1,1, 1,1):noBias()
    conv.weight:zero()
    for i=1,3 do
        conv.weight[{i,i}] = torch.Tensor{
            {0, strength, 0},
            {strength, 1 - 4*strength, strength},
            {0, strength, 0}
        }
    end
    return conv
end

function getIdentityConv()
    local conv = nn.SpatialConvolution(3,3, 3,3, 1,1, 1,1)
    conv.weight:zero()
    conv.bias:zero()
    for i=1,3 do
        conv.weight[{i,i,2,2}] = 1
    end
    return conv
end

-- simple interpolation.
function interpolate(vec_a, vec_b, weight_a)
    assert(weight_a >= 0  and weight_a <= 1, 'interp coefficient must be in [0,1]')
    local vec_interp = vec_a:clone()
    vec_interp:mul(weight_a):add(torch.mul(vec_b, 1-weight_a))
    return vec_interp
end

function augmentWithInterpolations(inputs)
    assert(inputs:nDimension()==4)
    assert(inputs:size(1)>1)
    local numSamples = inputs:size(1)
    local batch_size = numSamples
    for b=1,numSamples-1 do
        batch_size = batch_size + b
    end
    local C,H,W = inputs:size(2), inputs:size(3), inputs:size(4)
    local augmented_inputs = inputs.new():resize(batch_size,C,H,W)
    augmented_inputs[{{1,numSamples}}] = inputs
    local next_idx = numSamples+1
    for i=1,numSamples do
        for j=i+1,numSamples do
            local interp = interpolate(inputs[i], inputs[j], 0.5)
            augmented_inputs[next_idx] = interp
            next_idx = next_idx + 1
        end
    end
    return augmented_inputs
end
