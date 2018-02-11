
----
--given a time series, the attention model output a weight or a weighted vector for each time step of time series 

require 'nn'
local RNN = require 'model.my_RNN'
local LSTM = require 'model.my_LSTM'
local model_utils = require 'util.model_utils'
--require 'model.myMapTable'
require 'util/misc'

local Attention_weight = {}
Attention_weight.__index = Attention_weight

function Attention_weight.new(opt, if_interact)
  local self = {}
  setmetatable(self, Attention_weight)
  self:model(opt, if_interact)
  self.if_interact = if_interact
  return self
end

local function weight_model(opt,if_interact)

  local model = nn.Sequential()
  local map = nn.MapTable()
  
  
  local linearV, linearM
  if if_interact then
    linearV = nn.Linear(4*opt.att_size, 1)
    linearM = nn.Linear(4*opt.att_size, 4*opt.att_size)
  else
    linearV = nn.Linear(2*opt.att_size, 1)
    linearM = nn.Linear(2*opt.att_size, 2*opt.att_size)
  end
  local map_sub = nn.Sequential()
  map_sub:add(linearM)
  map_sub:add(nn.Tanh())
  map_sub:add(linearV)
  map:add(map_sub)
  model:add(map)
  model:add(nn.JoinTable(1, 1))
  model:add(nn.MulConstant(opt.att_sig_w))
  model:add(nn.Sigmoid())
  return model

end

function Attention_weight:model(opt, if_interact)
--  local net_all, net1, net2 = weight_model(opt, if_interact)
--  self.net_all = net_all
--  self.net1 = net1
--  self.net2 = net2
--  self.params_size = net_all:getParameters():nElement() + net1:getParameters():nElement() + net2:getParameters():nElement()
  local weight_net = weight_model(opt, if_interact)
  self.weight_net = weight_net
  self.params_size = weight_net:getParameters():nElement()
  self.h_weight = opt.att_sig_h
  print('number of parameters in the attention_weight model: ' .. self.params_size)
end

function Attention_weight:forward(hidden_v, summary)  
  local weight_net = self.weight_net
  local weights
  local hidden_s = {}
  if self.if_interact then
    for i = 1, #hidden_v do
      hidden_s[i] = torch.cat(hidden_v[i], torch.mul(summary,self.h_weight), 1)
    end
    weights = weight_net:forward(hidden_s)
  else
    weights = weight_net:forward(hidden_v)
  end
--  for i, k in pairs(hidden_v) do print(i, k) end
  
  return weights
end

function Attention_weight:backward(hidden_v, summary, gradOut, opt)
  local weight_net = self.weight_net
  local hidden_s = {}
  local grad_input
  if self.if_interact then
    for i = 1, #hidden_v do
      hidden_s[i] = torch.cat(hidden_v[i], torch.mul(summary,self.h_weight), 1)
    end
    grad_input = weight_net:backward(hidden_s, gradOut)
  else
    grad_input = weight_net:backward(hidden_v, gradOut)
  end
  
--  for i, k in pairs(grad_input) do print(i, k) end
  local grad_hidden = {}
  local grad_summary = nil
  if self.if_interact then
    local bit = grad_input[1]:size(1) / 2
    grad_summary = torch.zeros(bit)
    if opt.gpuid >= 0 then grad_summary = grad_summary:cuda() end
    for i = 1, #grad_input do
      grad_hidden[#grad_hidden+1] = grad_input[i]:sub(1,bit)
      grad_summary:add((grad_input[i]:sub(bit+1, -1)))
    end   
    grad_summary:mul(self.h_weight)
  else
    grad_hidden = grad_input
  end
  return grad_hidden, grad_summary
end

return Attention_weight


