require 'nn'
autograd = require 'autograd'

local module, parent = torch.class('nn.StyleGramMatrixLossModule', 'nn.Module')

function module:__init(strength, normalize)
    parent.__init(self)
    self.normalize = normalize or false
    self.strength = strength or 1
    self.target = nil
    self.loss = 0

    self.fun = self.GramMatrixLoss
    self.dfun = autograd(self.fun)
end

function module:clearState()
    self.dfun = nil
    return parent.clearState(self)
end

-- Input N x C x H x W
local gramMatrix = function(input)
    local N,C,H,W = input:size(1), input:size(2), input:size(3), input:size(4)
    local vecInput = input:view(N,C,H*W)
    local gramMatrix = torch.bmm(vecInput, torch.transpose(vecInput, 2,3))
    local output = gramMatrix / H / W
    return output
end

local squareErr = function(input, target)
    local buffer = input-target
    return torch.sum( torch.cmul(buffer, buffer) )
end

local sqFrobnorm = function(x)
    return torch.sum( torch.cmul(x,x) )
end

-- Input N x C x H x W
-- Target N x C x C
function module.GramMatrixLoss(input, target)
    local gmInput = gramMatrix(input)
    local sqErr = squareErr(gmInput, target)
    local sqNorm = sqFrobnorm(gmInput)
    local error = sqErr / sqNorm
    if input:nDimension()==4 then error = error / input:size(1) end
    return error
end

function module:setTarget(target_features)
    if target_features:nDimension() == 3 then
        local C,H,W = target_features:size(1), target_features:size(2), target_features:size(3)
        local target = gramMatrix(target_features:view(1,C,H,W))
        self.target = target:view(1,C,C)
    elseif target_features:nDimension() == 4 then
        local N,C,H,W = target_features:size(1), target_features:size(2), target_features:size(3), target_features:size(4)
        local target = gramMatrix(target_features)
        self.target = target:view(N,C,C)
    else
        error('Target must be 3D or 4D')
    end
    return self
end

function module:unsetTarget()
    self.target = nil
    return self
end

-- if # targets > 1 then assume # targets == # samples per batch
function module:updateOutput(input)
    self.output = input
    if self.target ~= nil then

        if input:nDimension() == 3 then
            local C,H,W = input:size(1), input:size(2), input:size(3)
            input = input:view(1,C,H,W)
        end
        assert(input:nDimension()==4)

        local N,C,H,W = input:size(1), input:size(2), input:size(3), input:size(4)
        assert(input:size(2) == self.target:size(2))

        if self.target:size(1) == 1 then
            self.match = self.target:expand(N,C,C)
        else
            self.match = self.target
        end

        self.loss = self.fun(input, self.target)
        self.loss = self.loss * self.strength
    end
    return self.output
end

function module:updateGradInput(input, gradOutput)
    if self.target ~= nil then
        local nInputDim = input:nDimension()
        if input:nDimension() == 3 then
            local C,H,W = input:size(1), input:size(2), input:size(3)
            input = input:view(1,C,H,W)
        end
        assert(input:nDimension()==4)

        local N,C,H,W = input:size(1), input:size(2), input:size(3), input:size(4)
        assert(input:size(2) == self.target:size(2))

        if self.target:size(1) == 1 then
            self.match = self.target:expand(N,C,C)
        else
            self.match = self.target
        end

        if self.dfun == nil then
            self.dfun = autograd(self.fun)
        end

        self.gradInput = self.dfun(input, self.target)

        if self.normalize then
            self.gradInput:div(torch.norm(self.gradInput, 1) + 1e-8)
        end

        if nInputDim == 3 then
            self.gradInput = self.gradInput:view(C,H,W)
        end

        self.gradInput:mul(self.strength)
        self.gradInput:add(gradOutput)
    else
        self.gradInput = gradOutput
    end
    return self.gradInput
end
