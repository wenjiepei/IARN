
  
--[[
The dot multiplicaton between the hidden representation of user and item
]]--

require 'nn'

local Bias_Term = {}

function Bias_Term:model(loader)
  local user_bias = torch.Tensor(#loader.user_X)
  local item_bias = torch.Tensor(#loader.item_X)
  local user_bias_grad = torch.zeros(#loader.user_X)
  local item_bias_grad = torch.zeros(#loader.item_X)
  Bias_Term.user_bias = user_bias
  Bias_Term.item_bias = item_bias
  Bias_Term.user_bias_grad = user_bias_grad
  Bias_Term.item_bias_grad = item_bias_grad
  Bias_Term.params_size = user_bias:nElement() + item_bias:nElement()
  Bias_Term.user_bias_size = user_bias:nElement()
  Bias_Term.item_bias_size = item_bias:nElement()
  user_bias:copy(loader.user_mean_rate):div(2)
  item_bias:copy(loader.item_mean_rate):div(2)
end

function Bias_Term:forward(user_ind, item_ind)
  local user_bias = Bias_Term.user_bias
  local item_bias = Bias_Term.item_bias
  local out = user_bias[user_ind] + item_bias[item_ind]
  return out
end

function Bias_Term:backward(user_ind, item_ind, gradout)
  local user_bias_grad = Bias_Term.user_bias_grad
  local item_bias_grad = Bias_Term.item_bias_grad
  user_bias_grad[user_ind] = user_bias_grad[user_ind]+gradout:squeeze()
  item_bias_grad[item_ind] = item_bias_grad[item_ind]+gradout:squeeze() 
end

function Bias_Term:parameters()
  return {Bias_Term.user_bias, Bias_Term.item_bias}, {Bias_Term.user_bias_grad, Bias_Term.item_bias_grad}
end

return Bias_Term

