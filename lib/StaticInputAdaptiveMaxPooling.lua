local module, parent = torch.class('nn.StaticInputAdaptiveMaxPooling', 'nn.Module')

function module:__init()
    self.module = nn.SpatialAdaptiveMaxPooling()
    self.output = torch.Tensor()
    self.gradInput = torch.Tensor()
    self._type = self.output:type()
end

function module:setStaticInput()
    local target = self.output
    if target:nDimension() == 4 then
        assert(target:size(1)==1, 'Only support non-batch static target.')
        local C,H,W = target:size(2), target:size(3), target:size(4)
        target = target:view(C,H,W)
    end
    assert(target:nDimension()==3, 'Only support spatial targets.')

    self.static_input = target:clone()
    self.C = target:size(1)
end

function module:unsetStaticInput()
    self.static_input = nil
    self.C = nil
end

function module:updateOutput(input)
    self.output = input
    if self.static_input ~= nil then
        local nInputDim = input:nDimension()
        if input:nDimension() == 3 then
            local C,H,W = input:size(1), input:size(2), input:size(3)
            input = input:view(1,C,H,W)
        end
        assert(input:nDimension()==4, 'Input must be 3D or 4D (batch).')
        assert(input:size(2) == self.C, 'Error: static input and input do not have the same number of channels.')

        self.H, self.W = input:size(3), input:size(4)
        self.module.H = self.H
        self.module.W = self.W

        self.output = self.module:forward(self.static_input):view(1,self.C, self.H, self.W):expandAs(input)

        if nInputDim == 3 then
            self.output = self.output[1]
        end
    end
    return self.output
end

function module:updateGradInput(input, gradOutput)
    self.gradInput:typeAs(input):resizeAs(input):zero()
    return self.gradInput
end