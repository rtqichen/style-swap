require 'nn'

randnet = {}

function randnet.nninit(module)
    module.weight:uniform(-1/math.sqrt(3), 1/math.sqrt(3))
    return module
end

--[[
Single layer of convolutions with kernel sizes of
3, 5, 7, 11, 15, 23, 37, 55
]]
function randnet.multiscale_conv(width)
    width = width or 128
    --local kSize = {3, 5, 7, 11, 15, 23, 37, 55}
    local kSize = {3, 5, 7, 11, 15, 23, 37}

    local net = nn.Sequential()
    local concat = nn.ConcatTable()
    for i=1,#kSize do
        local k = kSize[i]
        local pad = (k-1)/2
        local conv = randnet.nninit(nn.SpatialConvolution(3, width, kSize[i], kSize[i], 1,1, pad,pad):noBias())
        concat:add(nn.Sequential():add(conv):add(nn.ReLU(true)))
    end
    net:add(concat)
    net:add(nn.JoinTable(1,3))

    return net
end

--[[

]]
function randnet.random(width, k)
    width = width or 363
    k = k or 11
    local net = nn.Sequential()
    local pad = (k-1)/2
    net:add(randnet.nninit(nn.SpatialConvolution(3, width, k, k, 1,1, pad,pad):noBias()))
    net:add(nn.ReLU(true))
    return net
end

function randnet.multiscale_tconv(width)
    width = width or 128
    local kSize = {3, 5, 7, 11, 15, 23, 37, 55}

    local net = nn.Sequential()
    local concat = nn.ConcatTable()
    for i=1,#kSize do
        local k = kSize[i]
        local crop = (k-1)/2
        local tconv_crop = nn.Sequential()
        local tconv = nn.SpatialFullConvolution(3, width, kSize[i], kSize[i]):noBias()
        conv.weight:uniform(-1/math.sqrt(3), 1/math.sqrt(3))

        concat:add(nn.Sequential()
        :add(tconv)
        :add(nn.SpatialZeroPadding(-crop,-crop,-crop,-crop))
        :add(nn.ReLU(true)))
    end
    net:add(concat)
    net:add(nn.JoinTable(1,3))
    net:add(nn.ReLU(true))

    return net
end
