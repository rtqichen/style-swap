require 'nn'

-- require CUDA.
require 'cunn'
require 'cudnn'

require 'lib/ContentLossModule'
require 'lib/StyleGramMatrixLossModule'
require 'lib/StylePatchLossModule'
require 'lib/TVLossModule'
require 'helpers/preprocess'

local criterion, parent = torch.class('nn.ArtisticStyleLossCriterion', 'nn.Criterion')

function criterion:__init(cnn, layers, use_avg_pool, weights, target, only_pad_beginning, patch_size, patch_stride, normalize)
    parent.__init(self)

    layers = layers or {}
    layers.content = layers.content or {}
    layers.gramstyle = layers.gramstyle or {}
    layers.patchstyle = layers.patchstyle or {}

    weights = weights or {}
    weights.content = weights.content or 0
    weights.gramstyle = weights.gramstyle or 0
    weights.patchstyle = weights.patchstyle or 0
    weights.tv = weights.tv or 0

    if weights.gramstyle + weights.patchstyle + weights.tv + weights.content <= 0 then
        error('Weights must be provided / cannot all be zero.')
    end

    if weights.patchstyle <= 0 then
        layers.patchstyle = {}
    end
    if weights.gramstyle <= 0 then
        layers.gramstyle = {}
    end
    if weights.content <= 0 then
        layers.content = {}
    end

    if #layers.gramstyle + #layers.patchstyle + #layers.content == 0 then
        error('Must indicate layer(s) to attach loss modules to.')
    end

    use_avg_pool = use_avg_pool or false
    only_pad_beginning = only_pad_beginning or false
    patch_stride = patch_stride or 1
    patch_size = patch_size or 3
    if normalize==nil then
        normalize = true
    end

    local net = nn.Sequential()

    local patchstyle_layers = {}
    local gramstyle_layers = {}
    local content_layers = {}

    local next_gramstyle_idx = 1
    local next_patchstyle_idx = 1
    local next_content_idx = 1

    if weights.tv > 0 then
        local tv_mod = nn.TVLossModule(weights.tv):cuda()
        net:add(tv_mod)
    end
    local nop = function() end
    local pad_depth, required_padding = 0, 0
    for i=1,#cnn do
        if next_patchstyle_idx <= #layers.patchstyle or
            next_gramstyle_idx <= #layers.gramstyle or
            next_content_idx <= #layers.content then
            local layer = cnn:get(i)
            local name = layer.name
            if torch.type(layer) == 'nn.SpatialConvolution' or torch.type(layer) == 'cudnn.SpatialConvolution' then
                -- remove weight gradients.
                layer.accGradParameters = nop
                layer.gradWeight = nil
                layer.gradBias = nil
                -- setting padding to zero
                if only_pad_beginning then
                    layer.padW = 0
                    layer.padH = 0
                    required_padding = required_padding + math.pow(2,pad_depth)
                end
            end
            -- change max pooling to average pooling
            if torch.type(layer) == 'nn.SpatialMaxPooling' or torch.type(layer) == 'cudnn.SpatialMaxPooling' then
                layer:floor()
                pad_depth = pad_depth + 1
                if use_avg_pool then
                    local kW, kH, dW, dH = layer.kH, layer.kW, layer.dH, layer.dW
                    layer = nn.SpatialAveragePooling(kW, kH, dW, dH)
                end
            end

            net:add(layer)

            -- add loss modules
            if layers.patchstyle[next_patchstyle_idx] ~= nil and name == layers.patchstyle[next_patchstyle_idx] then
                local loss_module = nn.StylePatchLossModule(weights.patchstyle, normalize)
                net:add(loss_module)
                table.insert(patchstyle_layers, loss_module)
                next_patchstyle_idx = next_patchstyle_idx + 1
            end
            if layers.gramstyle[next_gramstyle_idx] ~= nil and name == layers.gramstyle[next_gramstyle_idx] then
                local loss_module = nn.StyleGramMatrixLossModule(weights.gramstyle, normalize)
                net:add(loss_module)
                table.insert(gramstyle_layers, loss_module)
                next_gramstyle_idx = next_gramstyle_idx + 1
            end
            if layers.content[next_content_idx] ~= nil and name == layers.content[next_content_idx] then
                local loss_module = nn.ContentLossModule(weights.content, normalize)
                net:add(loss_module)
                table.insert(content_layers, loss_module)
                next_content_idx = next_content_idx + 1
            end
        end
    end

    -- Error checking
    if next_patchstyle_idx < #layers.patchstyle then
        error('Could not find layer ' .. layers.patchstyle[next_patchstyle_idx])
    end
    if next_gramstyle_idx < #layers.gramstyle then
        error('Could not find layer ' .. layers.gramstyle[next_gramstyle_idx])
    end
    if next_content_idx < #layers.content then
        error('Could not find layer ' .. layers.content[next_content_idx])
    end

    -- Re-apply padding
    if only_pad_beginning then
        local padding_module = nn.SpatialReflectionPadding(required_padding, required_padding, required_padding, required_padding)
        if weights.tv > 0 then
            net:insert(padding_module, 2)
        else
            net:insert(padding_module, 1)
        end
    end

    -- Insert preprocessing images from [0,1]
    -- to whatever VGG-19 uses.
    net:insert(getPreprocessConv(), 1)

    -- Prepare
    self.net = cudnn.convert(net, cudnn):cuda()

    self.patchstyle_layers = patchstyle_layers
    self.gramstyle_layers = gramstyle_layers
    self.content_layers = content_layers

    self.dy = torch.CudaTensor()
    if target ~= nil then
        self:setTargets(target, patch_size, patch_stride)
    end
end

function criterion:setTargets(targets, patch_size, patch_stride)
    if targets.style == nil and targets.content == nil then
        error('Must provide either target.style or target.content images.')
    end
    self:unsetTargets()

    -- set latent responses from each style layer as the target
    if targets.style ~= nil then
        self:setStyleTarget(targets.style, patch_size, patch_stride)
    end

    if targets.content ~= nil then
        self:setContentTarget(targets.content)
    end
end

function criterion:setContentTarget(target)
    if #self.content_layers == 0 then return end
    if target == nil then
        error('Must provide target content image.')
    end
    assert(target:nDimension()==3 or target:nDimension()==4, 'Content target must be 3D or 4D (batch).')
    self.targets = self.targets or {}
    if target:type() ~= 'torch.CudaTensor' then
        target = target:cuda()
    end

    self.targets.content = target:clone()
    self.net:clearState()
    self.net:forward(self.targets.content)
    for i=1,#self.content_layers do
        local target_features = self.content_layers[i].output
        self.content_layers[i]:setTarget(target_features)
    end
end

function criterion:setStyleTarget(target, patch_size, patch_stride)
    if #self.patchstyle_layers + #self.gramstyle_layers <= 0 then
        return
    end
    if target == nil then
        error('Must provide target style image.')
    end
    assert(target:nDimension()==3 or target:nDimension()==4, 'Content target must be 3D or 4D (batch).')
    patch_size = patch_size or 3
    patch_stride = patch_stride or 1
    self.targets = self.targets or {}
    if target:type() ~= 'torch.CudaTensor' then
        target = target:cuda()
    end

    self.targets.style = target:clone()

    -- temporarily remove content targets, else the module
    -- may error out due to incorrect size.
    local content_targets = {}
    for i=1,#self.content_layers do
        content_targets[i] = self.content_layers[i].target
        self.content_layers[i].target = nil
    end

    self.net:clearState()
    self.net:forward(self.targets.style)
    for i=1,#self.patchstyle_layers do
        local target_features = self.patchstyle_layers[i].output
        self.patchstyle_layers[i]:setTarget(target_features, patch_size, patch_stride)
    end
    for i=1,#self.gramstyle_layers do
        local target_features = self.gramstyle_layers[i].output
        self.gramstyle_layers[i]:setTarget(target_features)
    end

    -- reset the content targets
    for i=1,#self.content_layers do
        self.content_layers[i].target = content_targets[i]
    end
end

function criterion:addStyleTarget(target)
    assert(#self.patchstyle_layers + #self.gramstyle_layers > 0, 'No style layers.')
    if target == nil then
        error('Must provide target style image.')
    end
    assert(target:nDimension()==3, 'Only supports 3D style target.')

    assert(#self.patchstyle_layers == 0,
        'Only gramstyle layers are supported for addStyleTarget, but there is one or more patch-style loss.')

    self.targets = self.targets or {}
    if target:type() ~= 'torch.CudaTensor' then
        target = target:cuda()
    end

    self.targets.style = target:clone()
    self.net:clearState()
    self.net:forward(self.targets.style)
    for i=1,#self.gramstyle_layers do
        local target_features = self.gramstyle_layers[i].output
        self.gramstyle_layers[i]:addTarget(target_features)
    end
end

function criterion:unsetTargets()
    for i=1,#self.patchstyle_layers do
        self.patchstyle_layers[i]:unsetTarget()
    end
    for i=1,#self.gramstyle_layers do
        self.gramstyle_layers[i]:unsetTarget()
    end
    for i=1,#self.content_layers do
        self.content_layers[i]:unsetTarget()
    end
end

--[[
Assumes input and target are both C x H x W images. (C=3)
Batch mode optional.
--]]
function criterion:updateOutput(input, targets)
    self.recompute_gradInput = true
    if not self.targets then self:setTargets(targets) end
    self.net:forward(input)
    -- accumulate losses from the style loss layers
    local styleLoss = 0
    local contentLoss = 0
    for _, mod in ipairs(self.patchstyle_layers) do
        styleLoss = styleLoss + mod.loss
    end
    for _, mod in ipairs(self.gramstyle_layers) do
        styleLoss = styleLoss + mod.loss
    end
    for _, mod in ipairs(self.content_layers) do
        contentLoss = contentLoss + mod.loss
    end
    self.styleLoss = styleLoss
    self.contentLoss = contentLoss
    self.output = styleLoss+contentLoss
    return self.output
end

function criterion:updateGradInput(input, targets)
    if self.recompute_gradInput then
        local dy = self.dy:resizeAs(self.net.output):zero()
        local grad = self.net:backward(input, dy)
        self.gradInput = grad:clone()
        -- reset targets
        if not self.targets then self:unsetTargets() end
    end
    self.recompute_gradInput = false
    return self.gradInput
end
