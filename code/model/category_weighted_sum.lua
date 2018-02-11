
  
--[[
The dot multiplicaton between the hidden representation of user and item
]]--

require 'nn'

local Item_Category = {}

local function basic_net(feature_dim)
  local unit_net = nn.Sequential()
  unit_net:add(nn.Linear(feature_dim, feature_dim, false))
  return unit_net
end

function Item_Category.model(loader, opt)
  local net = {}
  for i = 1, loader.max_category+1 do -- index from 0 
    net[#net+1] = basic_net(loader.feature_dim)
  end
  Item_Category.net = net
  local params_flat, _ = net[1]:getParameters()
  local param_size = params_flat:nElement() * (loader.max_category+1)
  Item_Category.param_size = param_size
  Item_Category.weight = opt.category_weight
  print('weight: ', Item_Category.weight)
  return net
end

function Item_Category.forward(category, x)
  local net = Item_Category.net
  local out = {}
  local w = 1 / Item_Category.weight
  for i = 1, category:nElement() do
    local ind = category[i]+1
    w = w * Item_Category.weight
    out[i] = torch.mul(net[ind]:forward(x), w)
  end
--  out[0] = x
  local addT = nn.CAddTable()
  local final_out = addT:forward(out)
  Item_Category.out = out
  Item_Category.final_out = final_out

  return final_out
end

function Item_Category.backward(category, x, gradout)
  local net = Item_Category.net
  local grad_in = {}
  local w = 1 / Item_Category.weight
  for i = 1, category:nElement() do
    local ind = category[i] + 1
    w = w * Item_Category.weight
    local temp_grad = torch.mul(gradout, w)
    grad_in[i] = net[ind]:backward(x, temp_grad)
  end
end

function Item_Category.free_memory()
  Item_Category.out = nil
  collectgarbage()
end

return Item_Category

