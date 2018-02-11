--
--
--local try = require 'model.trynn'
--local try1 = try.create()
--local try2 = try.create()
--print(try1.baba)
--print(try2.baba)
--print(try1.mama)
--print(try2.mama)
--try1:update(1,3)
--try2:update(2,4)
--
--print(try1.baba)
--print(try2.baba)
--print(try1.mama)
--print(try2.mama)
--
--require 'nn'
--
--local nn1 = nn.Linear(3,4,false)
--local nn2 = nn.Linear(3,4,false)
--print(nn1.weight)
--print(nn2.weight)
--nn1.weight:fill(1)
--print(nn1.weight)
--print(nn2.weight)
--
--local function bb(k)
--local aa = torch.Tensor({1,k})
--return aa
--end
--
--local aa = bb(4)
--local aa2 = bb(56)
--print(aa)
--print(aa2)
--

--local Category = require 'model.category_transformation' 
--local loader = {}
--loader.feature_dim = 2
--loader.max_category = 4
--local netc = Category.net(loader)
--print(netc[1]:get(1).weight)
--print(netc[2]:get(1).weight)
--print(netc[3]:get(1).weight)
--os.exit()
--local x = torch.Tensor({{1,2},{3,4}, {5,6}})
----local x = torch.Tensor({1,2})
--local category = torch.Tensor{1,3}
--local outv = Category.forward(category, x)
--print('haha')
--local temp = netc[2]:get(1).weight*x:select(1,1)
--local temp = torch.mm(netc[2]:get(1).weight, x:t())
--print(temp:t())
--temp = netc[4]:get(1).weight*temp
--print(temp:t())
--print(outv)

--local top_net = require 'model.Top_Net'
--
--local optt = {}
--optt.if_bidirection = 1
--optt.if_PReLU = 1
--optt.if_top_weighted_dot_product = 0
--local netss = top_net.dot_product_net(10, optt)
--print(netss)
--local nt = netss:get(1)
--print(nt)
--print(netss:get(1):get(1))

--local att_w = require 'model.attention_weight'
--
--local a = {}
--a[1] = torch.ones(4)
--a[2] = torch.ones(4)
--a[3] = torch.Tensor(4)
--
--local b = torch.Tensor(4)
--local opt = {}
--opt.att_size = 2
--opt.att_sig_w = 1
--local nett = att_w.new(opt, false)
--local f = nett:forward(a, b)
--print(f)
--local gradout = torch.ones(3)
--print(a[1]:size())
--local grad_h, grad_summary = nett:backward(a,b,gradout)
--for i, k in pairs(grad_h) do print(i, k) end
--print(grad_summary)

-- require 'torch'
--  local user_hidden, item_hidden, user_summary, item_summary, cate_item_x_set
--  local savefile = '/Users/wenjie/Documents/attention_recsys/code/attention_recsys/result/Sports/min-len_3/if-category_1__if-bias_0__user_IAGR__item_IAGR/att_1_64_sig-w_1__rnn_1_64_if-bidirection_0__dot-prod-weighted_0_if-PReLU_1/hidden_vec_dropout_0.00_negative_1.t7'
--    local hidden_v = torch.load(savefile)
--
--      user_hidden = hidden_v.user_hidden
--      user_summary = hidden_v.user_summary
--      item_hidden = hidden_v.item_hidden
--      item_summary = hidden_v.item_summary
--      cate_item_x_set = hidden_v.cate_item_x_set
--
--for i = 1, #user_hidden do
--  if #user_hidden[i] > 0 then
--    print(user_hidden[i][1]:size())
--    if user_hidden[i][1]:size(1) ~= 128 then 
--      error('~=128')
--    end
--  end
--end

require 'torch'

local a = torch.FloatTensor(1000,1000)
local b = torch.FloatTensor(1000,1000)

for i=1,10000 do
  local c = torch.mm(a,b)
  print(i)
end




