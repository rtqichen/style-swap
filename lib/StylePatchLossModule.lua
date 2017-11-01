local module, parent = torch.class('nn.StylePatchLossModule', 'nn.Module')

function module:__init(strength, normalize)
    parent.__init(self)
    self.strength = strength
    self.normalize = normalize or false

    self.target_patches = nil
    self.stitched_patches = nil
    self.conv_cc = nn.SpatialConvolution(1,1,1,1):noBias()
    self.conv_cc.gradWeight = nil

    self.conv_norm = nn.SpatialFullConvolution(1,1,1,1):noBias()
    self.conv_norm.gradWeight = nil

    self.crit = nn.MSECriterion()
end

function module._extract_patches(img, patch_size, stride)
    local nDim = 3
    assert(img:nDimension() == nDim, 'image must be of dimension 3.')
    
    kH, kW = patch_size, patch_size
    dH, dW = stride, stride
    local patches = img:unfold(2, kh, kW):unfold(3, kW, dW)
    n1, n2, n3, n4, n5 = patches:size(1), patches:size(2), patches:size(3), patches:size(4), patches:size(5)
    patches = patches:permute(2,3,1,4,5):contiguous():view(n2*n3, n1, n4, n5)

    return patches
end

-- This approach of extracting patches is much slower.
--[[
function module._extract_patches(img, patch_size, stride)
    local nDim = 3
    assert(img:nDimension() == nDim, 'image must be of dimension 3.')
    local C, H, W = img:size(nDim-2), img:size(nDim-1), img:size(nDim)
    local nH = math.floor( (H - patch_size)/stride + 1)
    local nW = math.floor( (W - patch_size)/stride + 1)

    -- extract patches
    local patches = torch.Tensor(nH*nW, C, patch_size, patch_size):typeAs(img)
    for i=1,nH*nW do
        local h = math.floor((i-1)/nW)  -- zero-index
        local w = math.floor((i-1)%nW)  -- zero-index
        patches[i] = img[{{},
        {1 + h*stride, 1 + h*stride + patch_size-1},
        {1 + w*stride, 1 + w*stride + patch_size-1}
        }]
    end

    return patches
end
]]

function module:setTarget(target_features, patch_size, patch_stride)
    assert(target_features:nDimension() == 3, 'Target must be 3D')
    local target_patches = self._extract_patches(target_features, patch_size, patch_stride)
    local npatches, C, kH, kW = target_patches:size(1), target_patches:size(2), target_patches:size(3), target_patches:size(4)
    assert(patch_size==kH and patch_size==kW, string.format('%d~=%d or %d~=%d', patch_size, kH, patch_size, kW))
    self.target_patches = target_patches:clone()

    -- convolution for computing cross correlation
    self.conv_cc.kH = kH
    self.conv_cc.kW = kW
    self.conv_cc.dH = patch_stride
    self.conv_cc.dW = patch_stride
    self.conv_cc.nInputPlane = C
    self.conv_cc.nOutputPlane = npatches
    local weight = target_patches:clone()
    -- normalize the patches to compute correlation
    for i=1,weight:size(1) do
        if torch.norm(weight[i],2) < 1e-6 then
            weight[i]:zero()
        else
            weight[i]:mul(1/torch.norm(weight[i],2))
        end
    end
    self.conv_cc.weight = weight

    -- convolution for computing normalization factor
    self.conv_norm.kH = kH
    self.conv_norm.kW = kW
    self.conv_norm.dH = patch_stride
    self.conv_norm.dW = patch_stride
    self.conv_norm.nInputPlane = 1
    self.conv_norm.nOutputPlane = 1
    self.conv_norm.weight = torch.Tensor(1,1,kH,kW):fill(1):typeAs(target_features)

    self.conv_cc:clearState()
    self.conv_norm:clearState()
end

function module:unsetTarget()
    self.target_patches = nil
end

function module:updateOutput(input)
    self.output = input

    if self.target_patches ~= nil then
        if input:nDimension() == 3 then
            local C,H,W = input:size(1), input:size(2), input:size(3)
            input = input:view(1,C,H,W)
        end
        assert(input:nDimension()==4)

        local N = input:size(1)
        local kH, kW = self.conv_cc.kH, self.conv_cc.kW
        local dH, dW = self.conv_cc.dH, self.conv_cc.dW
        local responses = self.conv_cc:forward(input)
        self.loss = 0
        self.last_targets = torch.Tensor():typeAs(input):resizeAs(input):zero()
        for b=1,N do
            local _, argmax = torch.max(responses[b],1)
            for h=1,argmax:size(2) do
                for w=1,argmax:size(3) do
                    local ind = argmax[{1,h,w}]
                    local target_patch = self.target_patches[ind]
                    local input_patch = input[{b, {},
                    {1 + (h-1)*dH, 1 + (h-1)*dH + kH-1},
                    {1 + (w-1)*dW, 1 + (w-1)*dW + kW-1}
                    }]
                    self.last_targets[{b, {},
                    {1 + (h-1)*dH, 1 + (h-1)*dH + kH-1},
                    {1 + (w-1)*dW, 1 + (w-1)*dW + kW-1}
                    }]:add(target_patch)
                    self.loss = self.loss + self.crit:forward(input_patch:reshape(input_patch:nElement()), target_patch:view(-1))
                end
            end
        end
        self.loss = self.loss / N
        self.loss = self.loss / input:nElement()
        self.loss = self.loss * self.strength
    end

    return self.output
end

--[[
TODO:

- instead of using nn.MSECriterion, which computes (x-a0) + (x-a1) + (x-a2) + ... + (x-ak),
calculate gradInput by summing up the target pixel values in the forward step and compute
(k*x - (a0+a1+a2+...ak))
--]]
-- function module:updateGradInput(input, gradOutput)
--     assert(input:isSameSizeAs(gradOutput))
--     if self.target_patches ~= nil then
--         local nInputDim = input:nDimension()
--         if input:nDimension() == 3 then
--             local C,H,W = input:size(1), input:size(2), input:size(3)
--             input = input:view(1,C,H,W)
--         end
--         assert(input:nDimension()==4)

--         local N, C = input:size(1), input:size(2)
--         local kH, kW = self.conv_cc.kH, self.conv_cc.kW
--         local dH, dW = self.conv_cc.dH, self.conv_cc.dW
--         local responses = self.conv_cc:forward(input)

--         self.gradInput = torch.Tensor():typeAs(input):resizeAs(input):zero()

--         for b=1,N do
--             local _, argmax = torch.max(responses[b],1)
--             for h=1,argmax:size(2) do
--                 for w=1,argmax:size(3) do
--                     local ind = argmax[{1,h,w}]
--                     local target_patch = self.target_patches[ind]
--                     local input_patch = input[{b, {},
--                     {1 + (h-1)*dH, 1 + (h-1)*dH + kH-1},
--                     {1 + (w-1)*dW, 1 + (w-1)*dW + kW-1}
--                     }]

--                     local grad = self.crit:backward(input_patch:reshape(input_patch:nElement()), target_patch:view(-1)):view(1,C,kH,kW)

--                     self.gradInput[{b, {},
--                     {1 + (h-1)*dH, 1 + (h-1)*dH + kH-1},
--                     {1 + (w-1)*dW, 1 + (w-1)*dW + kW-1}
--                     }]:add(grad)
--                 end
--             end

--             if self.normalize then
--                 self.gradInput[b]:div(torch.norm(self.gradInput[b], 1) + 1e-8)
--             end
--         end

--         if nInputDim == 3 then
--             local C,H,W = input:size(2), input:size(3), input:size(4)
--             self.gradInput = self.gradInput:view(C,H,W)
--         end

--         self.gradInput:mul(self.strength)
--         self.gradInput:add(gradOutput)
--     else
--         print(torch.type(self) .. ': No target set. Ignoring backward computation.')
--         self.gradInput = gradOutput
--     end
--     return self.gradInput
-- end

--[[
Equivalent to above code. Forces the use of MSE loss.
--]]
function module:updateGradInput(input, gradOutput)
    assert(input:isSameSizeAs(gradOutput))
    if self.target_patches ~= nil then
        local nInputDim = input:nDimension()
        if input:nDimension() == 3 then
            local C,H,W = input:size(1), input:size(2), input:size(3)
            input = input:view(1,C,H,W)
        end
        assert(input:nDimension()==4)

        local N, C, H, W = input:size(1), input:size(2), input:size(3), input:size(4)
        local kH, kW = self.conv_cc.kH, self.conv_cc.kW
        local dH, dW = self.conv_cc.dH, self.conv_cc.dW

        local oH = math.floor((H - kH) / dH + 1)
        local oW = math.floor((W - kW) / dW + 1)
        local ones = torch.Tensor():typeAs(input):resize(N,1,oH,oW):fill(1)
        local normalization = self.conv_norm:forward(ones):expandAs(input)

        self.gradInput = input:clone()
        self.gradInput:cmul(normalization):add(-self.last_targets)
        self.gradInput:mul(2/(kH*kW)/C)
        self.gradInput:div(input:nElement())

        if self.normalize then
            for b=1,N do
                self.gradInput[b]:div(torch.norm(self.gradInput[b], 1) + 1e-8)
            end
        end

        if nInputDim == 3 then
            local C,H,W = input:size(2), input:size(3), input:size(4)
            self.gradInput = self.gradInput:view(C,H,W)
        end

        self.gradInput:mul(self.strength)
        self.gradInput:add(gradOutput)
    else
        self.gradInput = gradOutput
    end
    return self.gradInput
end
