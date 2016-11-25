local module, parent = torch.class('nn.SoftThresholdLossModule', 'nn.Module')

--[[

Loss(x) = 1/2 * (x-max)^2   if x > max
        = 1/2 * (x-min)^2   if x < min

Gradient = 0      if min <= x <= max
         = x-max  if x > max
         = x-min  if x < min

--]]
function module:__init(min, max)
    parent.__init(self)

    assert(max >= min, string.format('max (%d) must be greater than min (%d)', max, min))

    self.min = min
    self.max = max
    self.output = torch.Tensor()
    self.gradInput = torch.Tensor()
end

function module:updateOutput(input)
    self.output = input
    return self.output
end

function module:updateGradInput(input, gradOutput)
    self.gradInput:resizeAs(input):zero()
    self.gradInput:add(torch.gt(input,self.max):type(self._type):cmul(torch.add(input, -self.max)))
    self.gradInput:add(torch.lt(input,self.min):type(self._type):cmul(torch.add(input, -self.min)))
    self.gradInput:add(gradOutput)
    return self.gradInput
end