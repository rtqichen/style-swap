-- Preprocess an image before passing it to a Caffe model.
-- We need to rescale from [0, 1] to [0, 255], convert from RGB to BGR,
-- and subtract the mean pixel.
function preprocess(img)
    local mean_pixel = torch.FloatTensor({103.939, 116.779, 123.68}):type(img:type())
    local perm = torch.LongTensor{3, 2, 1}
    img = img:index(1, perm):mul(255.0)
    mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
    img:add(-1, mean_pixel)
    return img
end

-- Undo the above preprocessing.
function deprocess(img)
    local mean_pixel = torch.FloatTensor({103.939, 116.779, 123.68}):type(img:type())
    mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
    img = img + mean_pixel
    local perm = torch.LongTensor{3, 2, 1}
    img = img:index(1, perm):div(255.0)
    return img
end

-- Computes the preprocess function using a nn.SpatialConvolution
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

-- Randomly crops the image so
-- the height and width are
-- divisible by some factor
function setDivBy(img, factor)
    local H, W = img:size(2), img:size(3)
    if H % factor ~= 0 or W % factor ~= 0 then
        local oheight = math.floor(H / factor)*factor       
        local owidth = math.floor(W / factor)*factor
        local y = torch.floor(torch.uniform(0, H-oheight+1))
        local x = torch.floor(torch.uniform(0, W-owidth+1))
        img = image.crop(img, x,y, x+owidth, y+oheight)
    end
    return img
end