require 'torch'
lapp = require 'pl.lapp'

opt = lapp[[
== Basic options ==
--style                 (default '')            File path to target image for style
--content               (default '')            File path to target image for content
--contentBatch          (default '')            Directory path to target images for content

== More Options ==
--maxContentSize        (default 640)           Maximum height and width for content image
--maxStyleSize          (default 512)           Maximum height and width for style image
--cpu                                           If set, uses CPU only
--save                  (default output)        Directory to save in
--saveOriginal                                  If set, saves the original image as well

== Advanced ==
--gpu                   (default 0)
--patchSize             (default 3)             Patch size for style swap [Higher = More Style Texture]
--patchStride           (default 1)             Patch stride for style swap operation
--pooling               (default 'max')         One of [avg|max]
--numSwap               (default 1)             Number of times to perform the style swap operation [Higher = More Style Contrast]

--decoder               (default '')            Path to a trained decoder
--optim                                         If set, decoder is only used for initialization and optimization still occurs

--learningRate          (default 0.05)          Learning rate for optimization
--init                  (default 'content')     How to initialize the generated image [random|content]
--tv                    (default 1e-7)          Weight for TV loss [Higher = Blur]
--layer                 (default 'relu3_1')     VGG layer to style swap on
--optimIter             (default 100)           Number of iterations for optimization
--printEvery            (default 50)            Print loss every so iterations
--saveLoss                                      If set, saves a table of loss values.
]]
print(opt)

if opt.style == '' then
    error('--style must be provided.')
end

if opt.content == '' and opt.contentBatch == '' then
    error('--content or --contentBatch must be provided.')
end

if not paths.filep(opt.style) then
    error('Style image ' .. opt.style .. ' does not exist.')
end

if opt.content ~= '' and not paths.filep(opt.content) then
    error('Content image ' .. opt.content .. ' does not exist.')
end

if opt.contentBatch ~= '' and not paths.dirp(opt.contentBatch) then
    error('Content directory ' .. opt.contentBatch .. ' does not exist.')
end

if opt.decoder ~= '' and not paths.filep(opt.decoder) then
    error('Decoder ' .. opt.decoder .. ' does not exist.')
end

print('Loading Lua modules...')

require 'nn'
if not opt.cpu then
    require 'cudnn'
    require 'cunn'
end
require 'loadcaffe'
require 'lib/ArtisticStyleLossCriterion'
require 'image'
require 'lib/ImageLoader'
require 'lib/NonparametricPatchAutoencoderFactory'
require 'lib/InstanceNormalization'
require 'optim'
require 'lib/MaxCoord'
require 'paths'
require 'image'

if not opt.cpu then
    cutorch.setDevice(opt.gpu+1)
end

vgg = loadcaffe.load('models/VGG_ILSVRC_19_layers_deploy.prototxt', 'models/VGG_ILSVRC_19_layers.caffemodel', 'nn')
for i=46,37,-1 do
    vgg:remove(i)
end

layers = {}
layers.content = {opt.layer}

weights = {}
weights.content = 1
weights.tv = opt.tv

use_avg_pooling = opt.pooling == 'avg'
criterion = nn.ArtisticStyleLossCriterion(vgg, layers, use_avg_pooling, weights, targets, false)
vgg = nil
collectgarbage()

if not opt.cpu then
    criterion.net = cudnn.convert(criterion.net, cudnn):cuda()
else
    criterion.net:float()
end

print(criterion.net)

if opt.decoder ~= '' then
    dec = torch.load(opt.decoder)
    decoder = nn.Sequential()
    decoder:add(nn.Unsqueeze(1)) -- add batch dim
    decoder:add(dec)
    decoder:add(nn.Squeeze(1))   -- remove batch dim
    if not opt.cpu then
        decoder:cuda()
    else
        decoder:float()
    end
    collectgarbage()
    print(dec)
end

local orig_window
local optim_window
local decoder_window

local optim_losses = {}

function synth(img)
    local x = img:clone()

    local sgdState = {
        learningRate = opt.learningRate
    }

    local losses = torch.Tensor(opt.optimIter)

    for i=1,opt.optimIter do
        function feval(x)
            local disp = x:clamp(0,1)

            local loss = criterion:forward(x)
            local loss_grad = criterion:backward(x)

            losses[i] = loss

            if i % opt.printEvery == 0 then
                print(string.format('%d, %e',i,loss))
            end

            return loss, loss_grad:view(-1)
        end
        optim.adam(feval, x, sgdState)
    end
    optim_losses[#optim_losses+1] = losses
    print('Done')
    return x
end

style_img = image.load(opt.style, 3)
if style_img:size(2) > opt.maxStyleSize or style_img:size(3) > opt.maxStyleSize then
    style_img = image.scale(style_img, opt.maxStyleSize)
end
if not opt.cpu then
    style_img = style_img:cuda()
else
    style_img = style_img:float()
end

criterion.targets = true    -- override behavior
criterion.net:forward(style_img)
style_latent = criterion.net.output:clone()

swap_enc, swap_dec = NonparametricPatchAutoencoderFactory.buildAutoencoder(style_latent, opt.patchSize, opt.patchStride, false, false, true)

swap = nn.Sequential()
swap:add(swap_enc)
swap:add(nn.MaxCoord())
swap:add(swap_dec)
swap:evaluate()

if not opt.cpu then
    swap:cuda()
else
    swap:float()
end

print(swap)

function swapTransfer(img, name)
    if opt.saveOriginal then
        image.save(opt.save .. '/' .. name, img)
    end

    if not opt.cpu then
        img = img:cuda()
    else
        img = img:float()
    end

    criterion:unsetTargets()
    criterion.net:forward(img)
    img_latent = criterion.net.output:clone()

    criterion.net:clearState()

    swap_latent = swap:forward(img_latent):clone()
    swap:clearState()

    if opt.decoder ~= '' then
        x = decoder:forward(swap_latent):clone()
        decoder:clearState()

        criterion.net.modules[#criterion.net.modules]:setTarget(swap_latent)

        if not opt.optim then
            local dec_loss = criterion:forward(x)
            optim_losses[#optim_losses+1] = torch.Tensor{dec_loss}
        else
            x = synth(x)
        end

    else
        local nUpsample = string.match(opt.layer, "(%d)_%d") -1
        local H,W = swap_latent:size(2)*math.pow(2,nUpsample), swap_latent:size(3)*math.pow(2,nUpsample)
        img = image.crop(img:float(), 0,0, W,H)
        if not opt.cpu then img = img:cuda() end
        if opt.init == 'random' then img:uniform() end
        criterion.net.modules[#criterion.net.modules]:setTarget(swap_latent)
        x = synth(img)
    end

    ext = paths.extname(name)
    image.save(opt.save .. '/' .. string.gsub(name, '.' .. ext, '_stylized.' .. ext), x)

    criterion.net:clearState()

    return x
end

print('Creating save folder at ' .. opt.save)
paths.mkdir(opt.save)

if opt.content ~= '' then
    img = image.load(opt.content, 3)
    local H,W = img:size(2), img:size(3)
    if H > opt.maxContentSize or W > opt.maxContentSize then
        img = image.scale(img, opt.maxContentSize)
    end

    name = paths.basename(opt.content)
    for i=1,opt.numSwap do
        img = swapTransfer(img, name)
    end
else
    imageLoader = ImageLoader(opt.contentBatch)
    imageLoader:setMaximumSize(opt.maxContentSize)

    for i=1, #imageLoader.files do
        img,name = imageLoader:next()
        for i=1,opt.numSwap do
            img = swapTransfer(img, name)
        end
    end
end

if opt.saveLoss then
    torch.save(opt.save .. '/loss.t7', optim_losses)
end
