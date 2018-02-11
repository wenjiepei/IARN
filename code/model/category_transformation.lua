
  
--[[
The dot multiplicaton between the hidden representation of user and item
]]--

require 'nn'

local Item_Category = {}

local function basic_net(feature_dim)
  local unit_net = nn.Sequential()
  unit_net:add(nn.Linear(feature_dim, feature_dim, false))
--  unit_net:add(nn.ReLU())
  unit_net:add(nn.Tanh())
  return unit_net
end

function Item_Category.model(loader)
  local net = {}
  for i = 1, loader.max_category+1 do -- index from 0 
    net[#net+1] = basic_net(loader.feature_dim)
  end
  Item_Category.net = net
  local params_flat, _ = net[1]:getParameters()
  local param_size = params_flat:nElement() * (loader.max_category+1)
  Item_Category.param_size = param_size
  return net
end

function Item_Category.forward(category, x)
  local net = Item_Category.net
  local out = {}
  for i = 1, category:nElement() do
    local ind = category[i]+1
    if i == 1 then
      out[i] = net[ind]:forward(x)
    else
      out[i] = net[ind]:forward(out[i-1])
    end
  end
  out[0] = x
  Item_Category.out = out

  return out[category:nElement()]:clone()
--  return out[category:nElement()]
end

function Item_Category.backward(category, x, gradout)
  local net = Item_Category.net
  local out = Item_Category.out
  local grad_in = {}
  for i = category:nElement(),1,-1 do
    local ind = category[i] + 1
    if i == category:nElement() then
      grad_in[i] = net[ind]:backward(out[i-1], gradout)
    elseif i == 1 then
      grad_in[i] = net[ind]:backward(x, grad_in[i+1])
    else
      grad_in[i] = net[ind]:backward(out[i-1], grad_in[i+1])
    end
  end
  return grad_in[1]
end

function Item_Category.free_memory()
  Item_Category.out = nil
  Item_Category.net = nil
  collectgarbage()
end

return Item_Category

