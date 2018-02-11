
  
--[[
The matrix factorization for user and item repectively
one vector for user and one vector for the item
]]--

require 'nn'
require 'torch'

local MF = {}

function MF.net(vec_size)
  local mf_net = nn.Sequential()
  local p = nn.ParallelTable()
  local user_v = nn.Linear(1, vec_size, false)
  local item_v = nn.Linear(1, vec_size, false)
  p:add(user_v)
  p:add(item_v)
  mf_net:add(p)
  mf_net:add(nn.DotProduct())
  return mf_net
end
--function MF.net(vec_size)
--  local user_v = nn.Linear(1, vec_size, false)
--  local item_v = nn.Linear(1, vec_size, false)
--  MF.user_v = user_v
--  MF.item_v = item_v
--  MF.vec_size = vec_size
--end
--
--function MF:forward()
--  local one = torch.ones(1)
--  local user_vec = MF.user_v:forward(one)
--  user_vec = user_vec:squeeze()
--  local item_vec = MF.item_v:forward(one):squeeze()
----  return {user_vec, item_vec}
--end

return MF

