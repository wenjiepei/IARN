
  
--[[
The dot multiplicaton between the hidden representation of user and item
]]--

require 'nn'

local Top_Net = {}

function Top_Net.dot_product_net(rnn_size, opt)

  local top_net = nn.Sequential()
  local p = nn.ParallelTable()
  local p1 = nn.Sequential()
  local h = rnn_size
  if opt.if_bidirection == 1 then
    h = h * 2
  end
  p1:add(nn.Linear(h, rnn_size))
  if opt.if_PReLU == 1 then
    p1:add(nn.PReLU())
  end
--  p1:add(nn.Linear(rnn_size, rnn_size))
  local p2 = nn.Sequential()
  p2:add(nn.Linear(h, rnn_size))
  if opt.if_PReLU == 1 then
    p2:add(nn.PReLU())
  end
--  p2:add(nn.Linear(rnn_size, rnn_size))
  p:add(p1)
  p:add(p2)
  top_net:add(p)
  if opt.if_weighted_dot_product == 1 then
    top_net:add(nn.CMulTable())
    top_net:add(nn.Linear(rnn_size, 1))
  else
    top_net:add(nn.DotProduct())
  end
  return top_net
end


return Top_Net