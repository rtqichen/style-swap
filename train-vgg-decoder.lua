require 'torch'
lapp = require 'pl.lapp'

opt = lapp[[
==== Required ====
--contentDir            (default '')                Content images for training
--styleDir              (default '')                Style images for training

==== Architecture ====
--activation            (default 'relu')            [relu|prelu|elu]
--instanceNorm                                      Replaces batchnorms with instance norm.
--subpixelConv          (default 0)                 Replaces upsampling with subpixel conv.
--tconv                                             Replaced convs with transposed convs
--upsample              (default 'nn')              [nn|bilinear]

==== Basic options ====
--maxIter               (default 80000)
--imageSize             (default 256)
--targetLayer           (default 'relu3_1')         Target hidden layer
--numSamples            (default 2)                 Batch size for training
--save                  (default 'vgginv')
--resume                (default '')                Model location
--gpu                   (default 0)

==== Optim ====
--learningRate          (default 1e-3)
--learningRateDecay     (default 1e-4)
--weightDecay           (default 0)
--normalize                                         Gradients at the loss function are normalized if enabled
--tv                    (default 1e-6)
--pixelLoss             (default 0)

==== Verbosity ====
--saveEvery             (default 500)
--printEvery            (default 10)
--display                                           Displays the training progress if enabled
--displayEvery          (default 20)
--displayAddr           (default '0.0.0.0')
--displayPort           (default 8000)
]]
print(opt)

if opt.contentDir == '' then
    error('--contentDir must be specified.')
end

if opt.styleDir == '' then
    error('--styleDir must be specified.')
end

require 'nn'
require 'cudnn'
require 'cunn'
require 'image'
require 'paths'
require 'optim'
nninit = require 'nninit'

require 'lib/ImageLoaderAsync'
require 'lib/TVLossModule'
require 'lib/NonparametricPatchAutoencoderFactory'
require 'lib/MaxCoord'
require 'lib/InstanceNormalization'
require 'helpers/utils'

if opt.display then
    display = require 'display'
    display.configure({hostname=opt.displayAddr, port=opt.displayPort})
end

paths.mkdir(opt.save)
torch.save(paths.concat(opt.save, 'options.t7'), opt)

cutorch.setDevice(opt.gpu+1)

---- Arguments ----
local decoderActivation
if opt.activation == 'relu' then
    decoderActivation = nn.ReLU
elseif opt.activation == 'prelu' then
    decoderActivation = nn.PReLU
elseif opt.activation == 'elu' then
    decoderActivation = nn.ELU
else
    error('Unknown activation option ' .. opt.activation)
end

---- Load VGG ----
require 'loadcaffe'
vgg = loadcaffe.load('models/VGG_ILSVRC_19_layers_deploy.prototxt', 'models/VGG_ILSVRC_19_layers.caffemodel', 'nn')

---- Extract Encoder and Create Decoder ----
enc = nn.Sequential()
for i=1,#vgg do
    local layer = vgg:get(i)
    enc:add(layer)
    local name = layer.name
    if name == opt.targetLayer then
        break
    end
end
if enc:get(#enc).name ~= opt.targetLayer then
    error('Could not find target layer ' .. opt.targetLayer)
end

if opt.resume ~= '' then
    dec = torch.load(opt.resume)
else
    dec = nn.Sequential()
    local lastLayerWidth
    for i=#enc,1,-1 do
        local layer = enc:get(i)

        if torch.type(layer):find('SpatialConvolution') then
            local nInputPlane, nOutputPlane = layer.nOutputPlane, layer.nInputPlane
            if opt.tconv then
                dec:add(nn.SpatialFullConvolution(nInputPlane, nOutputPlane, 3,3):init('weight', nninit.orthogonal, {gain = 'relu'}))
                dec:add(nn.SpatialZeroPadding(-1,-1,-1,-1))
            else
                dec:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1):init('weight', nninit.orthogonal, {gain = 'relu'}))
            end
            if opt.instanceNorm then
                dec:add(nn.InstanceNormalization(nOutputPlane))
            else
                dec:add(nn.SpatialBatchNormalization(nOutputPlane))
            end
            dec:add(decoderActivation())
            lastLayerWidth = nOutputPlane
        end
        if torch.type(layer):find('MaxPooling') then
            if opt.subpixelConv > 0 then
                dec:add(nn.SpatialConvolution(lastLayerWidth, lastLayerWidth*4, opt.subpixelConv,opt.subpixelConv, 1,1, (opt.subpixelConv-1)/2,(opt.subpixelConv-1)/2))
                dec:add(nn.PixelShuffle(2))
                dec:add(decoderActivation())
            else
                dec:add(nn.SpatialUpSamplingNearest(2))
            end
        end
    end

    dec:remove()
    dec:remove()
end

enc:insert(nn.TVLossModule(opt.tv), 1)
enc:insert(getPreprocessConv(), 1)

-- make sure to not cudnn the pooling layer.
enc = cudnn.convert(enc, cudnn):cuda()
dec = cudnn.convert(dec, cudnn):cuda()

print(enc)
print(dec)

---- Load Data ----
contentLoader = ImageLoaderAsync(opt.contentDir, opt.numSamples, {H=opt.imageSize, W=opt.imageSize})
styleLoader = ImageLoaderAsync(opt.styleDir, opt.numSamples, {H=opt.imageSize, W=opt.imageSize})

---- Criterion -----
criterion = nn.MSECriterion():cuda()
pixCriterion = nn.AbsCriterion():cuda()

---- Style Swap ----

function style_swap(content_latent, style_latent)
    local swap_enc, swap_dec = NonparametricPatchAutoencoderFactory.buildAutoencoder(style_latent, opt.patchSize, 1, false, false, true)

    local swap = nn.Sequential()
    swap:add(swap_enc)
    swap:add(nn.MaxCoord())
    swap:add(swap_dec)
    swap:evaluate()
    swap:cuda()

    local swap_latent =  swap:forward(content_latent):clone()
    swap:clearState()
    swap = nil
    collectgarbage()

    return swap_latent
end


---- Training -----
optim_state = {
    learningRate = opt.learningRate,
    learningRateDecay = opt.learningRateDecay,
    weightDecay = opt.weightDecay,
}

function maybe_print(trainLoss, timer)
    if optim_state.iterCounter % opt.printEvery == 0 then
        print(string.format('%7d\t\t%e\t%.2f\t%e',
        optim_state.iterCounter, trainLoss, timer:time().real, optim_state.learningRate))
        timer:reset()
    end
end

function maybe_display(inputs, reconstructions)
    if opt.display and (optim_state.iterCounter % opt.displayEvery == 0) then
        local batch_size = inputs:size(1)
        local disp = torch.cat(reconstructions:float(), inputs:float(), 1)
        if display_window then
            display.image(disp, {win=display_window, max=1, min=0})
        else
            display_window = display.image(disp, {max=1, min=0})
        end
    end
end

function maybe_save()
    if optim_state.iterCounter % opt.saveEvery == 0 then
        paths.mkdir(opt.save)
        local loc = paths.concat(opt.save, string.format('dec-%06d.t7', optim_state.iterCounter))
        torch.save(loc, cudnn.convert(dec:clearState():clone():float(), nn))
        torch.save(paths.concat(opt.save, 'enc.t7'), cudnn.convert(enc:clearState():clone():float(), nn))
    end
end

function train()
    optim_state.iterCounter = optim_state.iterCounter or 0

    local weights, gradients = dec:getParameters()
    print('Training...\tTrainErr\ttime\tLearningRate')
    local timer = torch.Timer()
    while optim_state.iterCounter < opt.maxIter do
        function feval(x)
            gradients:zero()
            optim_state.iterCounter = optim_state.iterCounter + 1

            local inputs = torch.FloatTensor(opt.numSamples*2, 3, opt.imageSize, opt.imageSize)
            inputs[{{1,opt.numSamples}}] = contentLoader:nextBatch()
            inputs[{{opt.numSamples+1,opt.numSamples*2}}] = styleLoader:nextBatch()
            inputs = inputs:cuda()

            local latent = enc:forward(inputs):clone()

            local C,H,W = latent:size(2), latent:size(3), latent:size(4)
            -- add more batch dimensions to account for style swaps
            latent:resize(opt.numSamples*2 + opt.numSamples^2, C,H,W)

            local add = 1
            for c=1,opt.numSamples do
                for s=1,opt.numSamples do
                    local content = latent[c]
                    local style = latent[opt.numSamples+s]
                    latent[opt.numSamples*2 + add] = style_swap(content, style)
                    add = add + 1
                end
            end

            ---- Dec -> Enc -> Loss

            local reconstructed_inputs = dec:forward(latent)

            local reconstructed_latent = enc:forward(reconstructed_inputs)
            local loss = criterion:forward(reconstructed_latent, latent)
            local enc_grad = criterion:backward(reconstructed_latent, latent)

            if opt.normalize then
                enc_grad:div(torch.norm(enc_grad, 1) + 1e-8)
            end

            local dec_grad = enc:backward(reconstructed_inputs, enc_grad)

            if opt.pixelLoss > 0 then
                local pixLoss = pixCriterion:forward(reconstructed_inputs[{{1,opt.numSamples*2}}], inputs)
                local dec_grad_pix = pixCriterion:backward(reconstructed_inputs[{{1,opt.numSamples*2}}], inputs)
                dec_grad_pix:mul(opt.pixelLoss)
                dec_grad[{{1,opt.numSamples*2}}]:add(dec_grad_pix)
            end

            dec:backward(latent, dec_grad)

            maybe_print(loss, timer)
            maybe_display(inputs, reconstructed_inputs)
            maybe_save()

            return loss, gradients
        end
        optim.adam(feval, weights, optim_state)
        collectgarbage()
    end
end

train()
