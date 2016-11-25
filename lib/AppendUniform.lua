local AppendUniform, parent = torch.class('nn.AppendUniform', 'nn.Module')

function AppendUniform:__init(n_append, lower_limit, upper_limit)
    parent.__init(self)
    self.n_append = n_append
    self.high = upper_limit
    self.low = lower_limit
end

function AppendUniform:updateOutput(input)
    local nInputDim = input:nDimension()
    if input:nDimension() == 3 then
        local C,H,W = input:size(1), input:size(2), input:size(3)
        input = input:view(1,C,H,W)
    end
    assert(input:nDimension()==4, 'Input must be 3D or 4D (batch).')

    local N,C,H,W = input:size(1), input:size(2), input:size(3), input:size(4)
    self.output = input.new():resize(N, C+self.n_append, H, W)
    self.output[{{},{1,C}}] = input
    self.output[{{},{C+1,C+self.n_append}}]:uniform(self.low, self.high)

    if nInputDim == 3 then
        self.output = self.output[1]
    end
    return self.output
end

function AppendUniform:updateGradInput(input, gradOutput)
    local nDim = gradOutput:nDimension()
    assert(nDim == 3 or nDim == 4, 'gradOutput must be 3D or 4D (batch).')
    local C = gradOutput:size(nDim-2)
    if gradOutput:nDimension() == 3 then
        self.gradInput = gradOutput[{{C-self.n_append,C}}]
    elseif gradOutput:nDimension() == 4 then
        self.gradInput = gradOutput[{{},{C-self.n_append,C}}]
    end
end