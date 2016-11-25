require 'torch'

local ImageLoaderAsync = torch.class('ImageLoaderAsync')

local threads = require 'threads'

local ImageLoader = {}
local ImageLoader_mt = { __index = ImageLoader }

---- Asynchronous image loader.
local result = {}
local H, W

function ImageLoaderAsync:__init(dir, batchSize, options)
    if not batchSize then
        error('Predetermined batch size is required for asynchronous loader.')
    end
    options = options or {}
    local n = options.n or 1

    -- upvalues
    H,W = options.H, options.W

    self.batchSize = batchSize
    self._type = 'torch.FloatTensor'

    -- initialize thread and its image loader
    self.threads = threads.Threads(n,
    function()
        imageLoader = ImageLoader:new(dir)
        if H ~= nil and W ~= nil then
            imageLoader:setWidthAndHeight(W,H)
        end
    end)

    -- get size
    self.threads:addjob(
    function() return imageLoader:size() end,
    function(size) result[1] = size end)
    self.threads:dojob()
    self._size = result[1]
    result[1] = nil

    -- add job
    for i=1,n do
        self.threads:addjob(self.__getBatchFromThread, self.__pushResult, self.batchSize)
    end
end

function ImageLoaderAsync:size()
    return self._size
end

function ImageLoaderAsync:type(type)
    if not type then
        return self._type
    else
        assert(torch.Tensor():type(type), 'Invalid type ' .. type .. '?')
        self._type = type
    end
    return self
end

function ImageLoaderAsync.__getBatchFromThread(batchSize)
    return imageLoader:nextBatch(batchSize)
end

function ImageLoaderAsync.__pushResult(batch)
    result[1] = batch
end

function ImageLoaderAsync:nextBatch()
    self.threads:addjob(self.__getBatchFromThread, self.__pushResult, self.batchSize)
    self.threads:dojob()
    local batch = result[1]
    result[1] = nil
    return batch:type(self._type)
end

---- Implementation of the actual image loader.

function ImageLoader:new(dir)
    require 'torch'
    require 'paths'
    require 'image'

    local imageLoader = {}
    setmetatable(imageLoader, ImageLoader_mt)
    local files = paths.dir(dir)
    local i=1
    while i <= #files do
        if not string.find(files[i], 'jpg$')
        and not string.find(files[i], 'png$')
        and not string.find(files[i], 'ppm$')then
            table.remove(files, i)
        else
            i = i +1
        end
    end
    imageLoader.dir = dir
    imageLoader.files = files
    imageLoader:rebatch()
    return imageLoader
end

function ImageLoader:size()
    return #self.files
end

function ImageLoader:rebatch()
    self.perm = torch.randperm(self:size())
    self.idx = 1
end

function ImageLoader:nextBatch(batchSize)
    local img = self:next()
    local batch = torch.FloatTensor(batchSize, 3, img:size(2), img:size(3))
    batch[1] = img
    for i=2,batchSize do
        batch[i] = self:next()
    end
    return batch
end

function ImageLoader:next()
    -- load image
    local img = nil
    local name
    local numErr = 0
    while true do
        if self.idx > self:size() then self:rebatch() end
        local i = self.perm[self.idx]
        self.idx = self.idx + 1
        name = self.files[i]
        local loc = paths.concat(self.dir, name)
        local status,err = pcall(function() img = image.load(loc,3,'float') end)
        if status then
            if self.verbose then print('Loaded ' .. self.files[i]) end
            break
        else
            io.stderr:write('WARNING: Failed to load ' .. loc .. ' due to error: ' .. err .. '\n')
        end
    end

    -- preprocess
    local H, W = img:size(2), img:size(3)
    if self.W and self.H then
        img = image.scale(img, self.W, self.H)
    elseif self.len then
        img = image.scale(img, self.len)
    elseif self.max_len then
        if H > self.max_len or W > self.max_len then
            img = image.scale(img, self.max_len)
        end
    end

    H, W = img:size(2), img:size(3)
    if self.div then
        local Hc = math.floor(H / self.div) * self.div
        local Wc = math.floor(W / self.div) * self.div
        img = self:_randomCrop(img, Hc, Wc)
    end

    if self.bnw then
        img = image.rgb2yuv(img)
        img[2]:zero()
        img[3]:zero()
        img = image.yuv2rgb(img)
    end

    collectgarbage()
    return img, name, numErr
end

---- Optional preprocessing

function ImageLoader:setVerbose(verbose)
    verbose = verbose or true
    self.verbose = verbose
end

function ImageLoader:setWidthAndHeight(W,H)
    self.H = H
    self.W = W
end

function ImageLoader:setFitToHeightOrWidth(len)
    assert(len ~= nil)
    self.len = len
    self.max_len = nil
end

function ImageLoader:setMaximumSize(max_len)
    assert(max_len ~= nil)
    self.max_len = max_len
    self.len = nil
end

function ImageLoader:setDivisibleBy(div)
    assert(div ~= nil)
    self.div = div
end

function ImageLoader:_randomCrop(img, oheight, owidth)
    assert(img:dim()==3)
    local H,W = img:size(2), img:size(3)
    assert(oheight <= H)
    assert(owidth <= W)
    local y = torch.floor(torch.uniform(0, H-oheight+1))
    local x = torch.floor(torch.uniform(0, W-owidth+1))
    local crop_img = image.crop(img, x,y, x+owidth, y+oheight)
    return crop_img
end

function ImageLoader:setBlackNWhite(bool)
    if bool then
        self.bnw = true
    else
        self.bnw = false
    end
end
