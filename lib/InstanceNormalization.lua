require 'nn'

--[[
Copied from https://github.com/jcjohnson/fast-neural-style .
----------------------------------------------------------------
Implements instance normalization as described in the paper

Instance Normalization: The Missing Ingredient for Fast Stylization
Dmitry Ulyanov, Andrea Vedaldi, Victor Lempitsky
https://arxiv.org/abs/1607.08022
This implementation is based on
https://github.com/DmitryUlyanov/texture_nets
]]

local InstanceNormalization, parent = torch.class('nn.InstanceNormalization', 'nn.Module')

function InstanceNormalization:__init(nOutput, eps)
    parent.__init(self)

    self.eps = eps or 1e-5

    self.nOutput = nOutput
    self.prev_N = -1

    self.weight = torch.Tensor(nOutput):uniform()
    self.bias = torch.Tensor(nOutput):zero()
    self.gradWeight = torch.Tensor(nOutput)
    self.gradBias = torch.Tensor(nOutput)
end


function InstanceNormalization:updateOutput(input)
    local N,C,H,W
    if input:nDimension() == 3 then
        N,C,H,W = 1, input:size(1), input:size(2), input:size(3)
    elseif input:nDimension() == 4 then
        N, C = input:size(1), input:size(2)
        H, W = input:size(3), input:size(4)
    end
    assert(C == self.nOutput)

    if N ~= self.prev_N or (self.bn and self:type() ~= self.bn:type()) then
        self.bn = nn.SpatialBatchNormalization(N * C, self.eps)
        self.bn:type(self:type())
        self.prev_N = N
    end

    -- Set params for BN
    self.bn.weight:repeatTensor(self.weight, N)
    self.bn.bias:repeatTensor(self.bias, N)

    local input_view = input:view(1, N * C, H, W)
    self.bn:training()
    self.output = self.bn:forward(input_view):viewAs(input)

    return self.output
end


function InstanceNormalization:updateGradInput(input, gradOutput)
    local N,C,H,W
    if input:nDimension() == 3 then
        N,C,H,W = 1, input:size(1), input:size(2), input:size(3)
    elseif input:nDimension() == 4 then
        N, C = input:size(1), input:size(2)
        H, W = input:size(3), input:size(4)
    end
    assert(self.bn)

    local input_view = input:view(1, N * C, H, W)
    local gradOutput_view = gradOutput:view(1, N * C, H, W)

    self.bn.gradWeight:zero()
    self.bn.gradBias:zero()

    self.bn:training()
    self.gradInput = self.bn:backward(input_view, gradOutput_view):viewAs(input)

    self.gradWeight:add(self.bn.gradWeight:view(N, C):sum(1))
    self.gradBias:add(self.bn.gradBias:view(N, C):sum(1))
    return self.gradInput
end


function InstanceNormalization:clearState()
    self.output = self.output.new()
    self.gradInput = self.gradInput.new()
    if self.bn then self.bn:clearState() end
end
