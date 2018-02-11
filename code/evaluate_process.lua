
require 'image'

local path = require 'pl.path'
local AUC_EER = require 'util/my_AUC_EER_calculation'
require 'util.misc'
local data_loader = require 'util.data_loader'
local model_utils = require 'util.model_utils'
local define_my_model = require 'model.define_my_model'
local table_operation = require 'util/table_operation'
local matio = require 'matio'
local evaluate_process = {}

--- preprocessing helper function
local function prepro(opt, x)
  if opt.gpuid >= 0 and opt.opencl == 0 then -- ship the input arrays to GPU
    x = x:float():cuda()
  end
  return x
end


local function save_attention_weight(set_name, user_weights, item_weights, predictions, opt, loader)
  local savefile = string.format('att_weight_dropout_%1.2f_negative_%d', opt.dropout, opt.if_negative)
  local pa = path.join(opt.current_result_dir, savefile )
  if not path.exists(pa) then lfs.mkdir(pa) end
  local temp_file = io.open(string.format( '%s/%s_weight.txt', pa, set_name), 'w')
  local nn, pair_index, true_T
  if set_name == 'train' then
    nn = predictions:nElement()
    pair_index = loader.train_pairs
    true_T = loader.train_T
  elseif set_name == 'validation' then
    nn = predictions:nElement()
    pair_index = loader.validation_pairs
    true_T = loader.train_T
  else
    nn = predictions:nElement()
    pair_index = loader.test_pairs
    true_T = loader.test_T
  end
  for i = 1, nn do
    local pairs = pair_index[i]
    temp_file:write(string.format('%5d, user %-5d, item %-5d, true_y %d, predict_y %-2.2f \n', i, pairs[1], pairs[2], true_T[i]:squeeze(), predictions[i]))
    local w_u = ""
    local w_item = ""
    for k = 1, user_weights[i]:nElement() do
      w_u = w_u .. string.format('%1.2f  ', user_weights[i][k])
    end
    for k = 1, item_weights[i]:nElement() do
      w_item = w_item .. string.format('%1.2f  ', item_weights[i][k])
    end
    temp_file:write(string.format('%s\n', w_u))
    temp_file:write(string.format('%s\n', w_item))
--    temp_file:write('\n\n')
  end
  temp_file:close()
end


--- calculate the hidden representation for all the users and items, for IARN
local function extract_hidden_representation_IARN(model, type, input_x, opt, category)
  -- decode the model and parameters
  local category_transform = model.category_transform
  local user_attention = model.user_attention
  local user_att_weight_net = model.user_att_weight
  local item_attention = model.item_attention
  local item_att_weight_net = model.item_att_weight
  local rnn_user = model.rnn_user
  local rnn_item = model.rnn_item
  local top_net = model.top_net
  local MF_net = model.MF_net
  local bias_term = model.Bias_Term
  local criterion = model.criterion 
  local params_flat = model.params_flat

  local x_length = input_x:size(2)

  local hidden_z_value, hidden_summary
  local cate_item_x = nil
  local rnn_net_output
  if type == 'user' then
    -- perform the forward pass for attention model
    hidden_z_value, hidden_summary = user_attention:forward(input_x, opt, 'test')
--    if hidden_z_value[1]:size(1) > 128 then 
--      error(': >128')
--    end
--    hidden_z_value = table_operation.shallowCopy(hidden_z_value)
  elseif type == 'item' then
    -- perform the forward pass for the category transform for item time series
    cate_item_x = nil
    if opt.if_category == 1 and category:nElement() > 0 then
      local cate_trans = category_transform.forward(category, input_x:t())
      cate_item_x = cate_trans:clone():t()
    else
      cate_item_x = input_x
    end
    hidden_z_value, hidden_summary = item_attention:forward(cate_item_x, opt, 'test')
  else
    error('no such type!')
  end
  return hidden_z_value, hidden_summary, cate_item_x 
end

function evaluate_process.extract_all_hidden_representation_IARN(model, loader, opt)
  local savefile = string.format('%s/hidden_vec_dropout_%1.2f_negative_%d.t7', opt.current_result_dir, opt.dropout, opt.if_negative)
  if file_exists(savefile) then return end
  --- for user
  local user_hidden = {}
  local user_summary = torch.Tensor(#loader.user_X, 2*opt.rnn_size)
  if opt.gpuid >= 0 then user_summary = user_summary:cuda() end
  for i = 1, #loader.user_X do
    local hidden_v, hidden_summary
    if loader.remove_user_ind:eq(i):sum() >0 then
      hidden_v = {}
      hidden_summary = torch.zeros(2*opt.rnn_size)
      if opt.gpuid >= 0 then hidden_summary = hidden_summary:cuda() end
    else
      local type = 'user'
      local input_x = loader.user_X[i]
      hidden_v, hidden_summary = extract_hidden_representation_IARN(model, type, input_x, opt)
    end
    user_hidden[i] = hidden_v
    user_summary[i] = hidden_summary
    if i % 1000 == 0 then
      print(i, 'finished!')
    end
  end
  print('user hidden representation finished!')
  --- for item
  local item_hidden = {}
  local item_summary = torch.Tensor(#loader.item_X, 2*opt.rnn_size)
  if opt.gpuid >= 0 then item_summary = item_summary:cuda() end
  local cate_item_x_set = {}
  for i = 1, #loader.item_X do
    local hidden_v, hidden_summary, cate_item_x
    if loader.remove_item_ind:eq(i):sum() >0 then
      hidden_v = {}
      hidden_summary = torch.zeros(2*opt.rnn_size)
      if opt.gpuid >= 0 then hidden_summary = hidden_summary:cuda() end
    else
      local type = 'item'
      local input_x = loader.item_X[i]
      local category = nil
      if opt.if_category == 1 then
        category = loader.item_category[i]
      end
      hidden_v, hidden_summary, cate_item_x = extract_hidden_representation_IARN(model, type, input_x, opt, category)
    end
    item_hidden[i] = hidden_v
    item_summary[i] = hidden_summary
    cate_item_x_set[i] = cate_item_x
    if i % 1000 == 0 then
      print(i, 'finished!')
    end
  end
  print('item hidden representation finished!')
  ---save the results
  print('saving checkpoint to ' .. savefile)
  -- to save the space, we only save the parameter values in the model
  local hidden_v = {
    user_hidden = user_hidden,
    user_summary = user_summary,
    item_hidden = item_hidden,
    item_summary = item_summary,
    cate_item_x_set = cate_item_x_set
  } 
  torch.save(savefile, hidden_v)
end

--- calculate the hidden representation for all the users and items, for TAGM-TAGM, rnn-rnn and lstm-lstm
local function extract_hidden_representation_normal(model, type, input_x, opt, category)
  -- decode the model and parameters
  local category_transform = model.category_transform
  local user_attention = model.user_attention
  local item_attention = model.item_attention
  local user_att_weight_net = model.user_att_weight
  local item_att_weight_net = model.item_att_weight
  local rnn_user = model.rnn_user
  local rnn_item = model.rnn_item
  local top_net = model.top_net
  local MF_net = model.MF_net
  local bias_term = model.Bias_Term
  local criterion = model.criterion 
  local params_flat = model.params_flat

  local x_length = input_x:size(2)
  
  local att_weights, hidden_z_value, rnn_hidden, hidden_summary
  local rnn_net_output
  
  if type == 'user' then
    -- perform the forward pass for attention model
    if opt.user_module == 'TAGM' then
      hidden_z_value, hidden_summary = user_attention:forward(input_x, opt, 'test')
      hidden_summary = torch.zeros(1)
      if opt.gpuid >= 0 then hidden_summary = hidden_summary:cuda() end
      att_weights = user_att_weight_net:forward(hidden_z_value, hidden_summary)
    else
      att_weights = torch.ones(x_length)
      if opt.gpuid >= 0 then att_weights = att_weights:cuda() end
    end
    -- perform the forward for the rnn
    rnn_hidden = rnn_user:forward(input_x, att_weights, opt, 'test')
    -- perform the forward for the top-net module
    local top_net_sub = top_net:get(1):get(1)
    rnn_net_output = top_net_sub:forward(rnn_hidden)
  elseif type == 'item' then
    -- perform the forward pass for the category transform for item time series
    local cate_item_x = nil
    if opt.if_category == 1 and category:nElement() > 0 then
      local cate_trans = category_transform.forward(category, input_x:t())
      cate_item_x = cate_trans:t()
    else
      cate_item_x = input_x
    end
    if opt.item_module == 'TAGM' then
      hidden_z_value, hidden_summary = item_attention:forward(cate_item_x, opt, 'test')
      hidden_summary = torch.zeros(1)
      if opt.gpuid >= 0 then hidden_summary = hidden_summary:cuda() end
      att_weights = item_att_weight_net:forward(hidden_z_value, hidden_summary)
    else
      att_weights = torch.ones(x_length)
      if opt.gpuid >= 0 then att_weights = att_weights:cuda() end
    end
    -- perform the forward for the rnn
    rnn_hidden = rnn_item:forward(cate_item_x, att_weights, opt, 'test')
    -- perform the forward for the top-net module
    local top_net_sub = top_net:get(1):get(2)
    rnn_net_output = top_net_sub:forward(rnn_hidden)
  else
    error('no such type!')
  end
  return rnn_net_output
end

function evaluate_process.extract_all_hidden_representation_normal(model, loader, opt)
  local savefile = string.format('%s/hidden_vec_dropout_%1.2f_negative_%d.t7', opt.current_result_dir, opt.dropout, opt.if_negative)
  if file_exists(savefile) then return end
  --- for user
  local user_hidden = torch.Tensor(#loader.user_X, opt.rnn_size)
  if opt.gpuid >= 0 then user_hidden = user_hidden:cuda() end
  for i = 1, #loader.user_X do
    local hidden_v
    if loader.remove_user_ind:eq(i):sum() >0 then
      hidden_v = torch.zeros(opt.rnn_size)
      if opt.gpuid >= 0 then hidden_v = hidden_v:cuda() end
    else
      local type = 'user'
      local input_x = loader.user_X[i]
      hidden_v = extract_hidden_representation_normal(model, type, input_x, opt)
    end
    user_hidden[i] = hidden_v
    if i % 1000 == 0 then
      print(i, 'finished!')
    end
  end
  print('user hidden representation finished!')
  --- for item
  local item_hidden = torch.Tensor(#loader.item_X, opt.rnn_size)
  if opt.gpuid >= 0 then item_hidden = item_hidden:cuda() end
  for i = 1, #loader.item_X do
    local hidden_v
    if loader.remove_item_ind:eq(i):sum() >0 then
      hidden_v = torch.zeros(opt.rnn_size)
      if opt.gpuid >= 0 then hidden_v = hidden_v:cuda() end
    else
      local type = 'item'
      local input_x = loader.item_X[i]
      local category = loader.item_category[i]
      hidden_v = extract_hidden_representation_normal(model, type, input_x, opt, category)
    end
    item_hidden[i] = hidden_v
    if i % 1000 == 0 then
      print(i, 'finished!')
    end
  end
  print('item hidden representation finished!')
  ---save the results
  print('saving hidden representation to ' .. savefile)
  -- to save the space, we only save the parameter values in the model
  local hidden_v = {
    user_hidden = user_hidden,
    item_hidden = item_hidden
  } 
  torch.save(savefile, hidden_v)
end

local function calculate_pair_dot_product_IARN(user_ind, item_ind, opt, loader, model, 
    user_hidden, item_hidden, user_summary, item_summary, cate_item_x)
  -- decode the model and parameters
  local category_transform = model.category_transform
  local user_attention = model.user_attention
  local item_attention = model.item_attention
  local user_att_weight_net = model.user_att_weight
  local item_att_weight_net = model.item_att_weight
  local rnn_user = model.rnn_user
  local rnn_item = model.rnn_item
  local top_net = model.top_net
  local MF_net = model.MF_net
  local bias_term = model.Bias_Term
  local criterion = model.criterion 
  local params_flat = model.params_flat

  local user_x = loader.user_X[user_ind]
  local item_x = loader.item_X[item_ind]
--  print('user_hidden:', #user_hidden, user_hidden[1]:size())
--  print('item summary: ', item_summary:size())
  local user_att_weights = user_att_weight_net:forward(user_hidden, item_summary)
  local item_att_weights = item_att_weight_net:forward(item_hidden, user_summary)
  -- perform the forward for the rnn_user and rnn_item
  local user_hidden = rnn_user:forward(user_x, user_att_weights, opt, 'test')
  local item_hidden = rnn_item:forward(cate_item_x, item_att_weights, opt, 'test')
  -- perform the forward for the top-net module
  local rnn_net_output = top_net:forward({user_hidden, item_hidden})

  return rnn_net_output:squeeze()
end

--- calculate the normalized discounted cumulative gain
local function calcuate_NDCG_AUC(opt, loader, model)
  local user_hidden, item_hidden, user_summary, item_summary, cate_item_x_set
  local savefile = string.format('%s/hidden_vec_dropout_%1.2f_negative_%d.t7', opt.current_result_dir, opt.dropout, opt.if_negative) 
  if file_exists(savefile) then
    local hidden_v = torch.load(savefile)
    if opt.user_module == 'IARN' then
      user_hidden = hidden_v.user_hidden
      user_summary = hidden_v.user_summary
      item_hidden = hidden_v.item_hidden
      item_summary = hidden_v.item_summary
      cate_item_x_set = hidden_v.cate_item_x_set
    else
      user_hidden = hidden_v.user_hidden
      item_hidden = hidden_v.item_hidden
    end
  else
    error('cannot find hidden representation!')
  end
  -- get the user index mentioned in test set
  local test_user_index, test_user_reverse_index
  local dd = {}
  if opt.data_set == 'MovieLens' or opt.data_set == 'Netflix' then
    --    test_user_index = torch.ones(loader.test_pairs:size(1)):fill(-1)
    --    test_user_reverse_index = torch.zeros(#loader.user_X):fill(-5)
    --    local ind = 0
    --    local vec = loader.test_pairs:select(2,1)
    --    for i = 1,  vec:nElement() do
    --      local v = vec[i]
    --      if ind == 0 or torch.sum(test_user_index:sub(1, ind):eq(v+1)) == 0 then
    --        ind = ind + 1
    --        test_user_index[ind] = v+1
    --        test_user_reverse_index[v+1] = ind
    --      end
    --    end
    --    test_user_index = test_user_index:sub(1, ind)

    local index_file = '../../data/test_users_five_hundreds.t7'
    local all_index = torch.load(index_file)
    all_index = (all_index.test_users_five_hundreds):t()
    dd['MovieLens'] = 1
    dd['Netflix'] = 2
    test_user_index = all_index[dd[opt.data_set]]
  else
    local index_file = '../../data/test_users_one_thousand.t7'
    local all_index = torch.load(index_file)
    all_index = (all_index.test_users_one_thousand):t()
    dd['Clothing'] = 1
    dd['Sports'] = 2
    dd['Electronics'] = 3
    dd['Home'] = 4
    dd['Netflix'] = 5
    test_user_index = all_index[dd[opt.data_set]]
  end
  test_user_reverse_index = torch.zeros(#loader.user_X):fill(-5)
  if opt.gpuid >= 0 then test_user_reverse_index = test_user_reverse_index:cuda() end
  local new_test_user_index = {}
  local indtemp = 0
  for i = 1, test_user_index:size(1) do
    local inv = test_user_index[i]+1
    if loader.remove_user_ind:eq(inv):sum() == 0 then
      new_test_user_index[#new_test_user_index+1] = inv
      test_user_reverse_index[inv] = #new_test_user_index
      --        test_user_index[i] = inv
      --        test_user_reverse_index[inv] = i
    else
      ------remove that item!!!!
      print('warning: the user ind is in the removed list: ', inv)
    end
  end
  new_test_user_index = torch.Tensor(new_test_user_index)
  print('test user size: ', new_test_user_index:size(1))
  test_user_index = new_test_user_index

  
  -- get the item index mentioned in training and test set
  local test_item_index = torch.ones(#loader.item_X):fill(-1)
  local test_item_reverse_index = torch.zeros(#loader.item_X):fill(-5)
  local ind = 0
  local vec = loader.test_pairs:select(2,2)
  for i = 1,  vec:nElement() do
    local v = vec[i]
    if ind == 0 or torch.sum(test_item_index:sub(1, ind):eq(v+1)) == 0 then
      ind = ind + 1
      test_item_index[ind] = v+1
      test_item_reverse_index[v+1] = ind
    end
  end
  vec = loader.train_pairs:select(2,2)
  for i = 1,  vec:nElement() do
    local v = vec[i]
    if torch.sum(test_item_index:sub(1, ind):eq(v+1)) == 0 then
      ind = ind + 1
      test_item_index[ind] = v+1
      test_item_reverse_index[v+1] = ind
    end
  end
  test_item_index = test_item_index:sub(1, ind)
  print('test item size: ', test_item_index:size(1))
  
  -- and the groundtruth
  local groundt = torch.zeros(test_user_index:size(1), test_item_index:size(1))
  if opt.gpuid >= 0 then groundt = groundt:cuda() end
  for i = 1,loader.test_pairs:size(1) do
    local us = loader.test_pairs[i][1]
    us = test_user_reverse_index[us+1]
    local it = loader.test_pairs[i][2]
    it = test_item_reverse_index[it+1]
    if it > 0 and us > 0 then
--      print('us:', us, 'it:', it)
      groundt[us][it] = 1
    elseif it < 1 then
      error('not it > 0 and us > 0')
    end
  end
  print(groundt:sum(2):eq(0):sum())
  print(groundt:sum())
--os.exit()
  for i = 1,loader.train_pairs:size(1) do
    local us = loader.train_pairs[i][1]
    us = test_user_reverse_index[us+1]
    if us>0 then
      local it = loader.train_pairs[i][2]
      it = test_item_reverse_index[it+1]
      if it > 0 then
        groundt[us][it] = -1
      else
        error('it < 0')
      end
    end
  end

  local mm_p
  local timer = torch.Timer()
  local time_s = timer:time().real
  if opt.user_module == 'IARN' then
    local savefile = string.format('%s/pair_IARN_product_dropout_%1.2f_negative_%d.t7', opt.current_result_dir, opt.dropout, opt.if_negative)
    if not file_exists(savefile) then 
      mm_p = torch.zeros(test_user_index:nElement(), test_item_index:nElement())
      if opt.gpuid >= 0 then mm_p = mm_p:cuda() end
      local indd = 0
      print('begin to calculate dot product for all pairs with IARN-IARN...')
      local total_n = test_user_index:size(1) * test_item_index:size(1)
      for m = 1, test_user_index:size(1) do
        for n =1, test_item_index:size(1) do
          local user_ind = test_user_index[m]
          local item_ind = test_item_index[n]
          local user_hidden_v = user_hidden[user_ind]
          local item_hidden_v = item_hidden[item_ind]
          local user_summary_v = user_summary[user_ind]
          local item_summary_v = item_summary[item_ind]
          local cate_item_x = cate_item_x_set[item_ind]
          mm_p[m][n] = calculate_pair_dot_product_IARN(user_ind, item_ind, opt, loader, 
            model,user_hidden_v, item_hidden_v, user_summary_v, item_summary_v, cate_item_x)
          indd = indd + 1
          if indd % 10000 == 0 then 
            print(indd .. ' / ' .. total_n, 'finished!')
            collectgarbage()
            local time_se = timer:time().real
            local times = time_se - time_s
            print('elasped time: ', times)
          end
        end
      end
      ---save the results
      print('saving pair product to ' .. savefile)
      -- to save the space, we only save the parameter values in the model
      local pair_product = {
        mm_p = mm_p
      } 
      torch.save(savefile, pair_product)
    else
      local pair_product = torch.load(savefile)
      mm_p = pair_product.mm_p
    end
  else  
    local user_h = user_hidden:index(1, test_user_index:long())
    local item_h = item_hidden:index(1, test_item_index:long())
    mm_p = torch.mm(user_h, item_h:t())
  end
  local time_e = timer:time().real
  local times = time_e - time_s
  print(mm_p:size())
  print('time of computing paired scores: ', times)
  
  local sort_p, sort_ind = torch.sort(mm_p, 2, true)
  if opt.gpuid >= 0 then
     sort_p = sort_p:cuda()
     sort_ind = sort_ind:cuda() 
  end
  local time_e2 = timer:time().real
  print('time of sorting: ', time_e2-time_e) 
  local function NDCG_AUC(i, top_k)
--    print(i)
    local dcg = 0
    local precision = torch.zeros(#top_k)
    local recall = torch.zeros(#top_k)
    if opt.gpuid >= 0 then
     precision = precision:cuda()
     recall = recall:cuda() 
    end
    local sort_i = sort_ind[i]
--    print('sort_mm', mm_p[i]:index(1, sort_i):sub(1,200))
    local temp = groundt[i]:index(1, sort_i)
--    print(temp:size(1))
    local indd = temp:ne(-1)
    local sort_g = temp[indd]
--    print(sort_g:size(1))
    local correct_pairs = 0
    local hits = 0
    local mean_average_precision = 0
    local pn = torch.sum(sort_g:eq(1))
    if opt.gpuid >= 0 then
     pn = pn:cuda() 
    end
    for k = 1, sort_g:nElement() do
      if sort_g[k] > 0 then
        dcg = dcg + 1 / (math.log(k+1)/math.log(2))
        hits = hits+1
        mean_average_precision = mean_average_precision + hits / k
      else
        correct_pairs = correct_pairs + hits
      end
    end
--    print(correct_pairs, pn, sort_g:nElement())
    local AUC = 0
    local idcg = 0
    local ndcg = 0
    if pn==0 then
      return ndcg, precision, recall, AUC, mean_average_precision
        --      error('pn==0')
    else
      AUC = correct_pairs / (pn*(sort_g:nElement()-pn))
      mean_average_precision = mean_average_precision / pn
      for m = 1, pn do
        idcg = idcg + 1 / (math.log(m+1)/math.log(2))
      end
    end
    local ndcg = dcg / idcg
--    print(dcg, idcg)
    -- calculate the precision
    local precision = torch.Tensor(#top_k)
    local recall = torch.Tensor(#top_k)
    for m = 1, #top_k do
      local K = top_k[m]
      local sort_p = sort_g:sub(1,K)
      local hit = torch.sum(sort_p)
--      print('hit:', hit)
      precision[m] = hit / K
      recall[m] = hit / pn
    end
    -- calculate the AUC
    for k = 1, sort_g:nElement() do
    end
    
    sort_g = nil
    return ndcg, precision, recall, AUC, mean_average_precision
  end
  
  local top_k = {5, 10, 20}
  local user_NDCG = torch.Tensor(test_user_index:size(1))
  local user_precision = torch.Tensor(test_user_index:size(1), #top_k)
  local user_recall = torch.Tensor(test_user_index:size(1), #top_k)
  local user_AUC = torch.Tensor(test_user_index:size(1))
  local user_map = torch.Tensor(test_user_index:size(1))
  if opt.gpuid >= 0 then
    user_NDCG = user_NDCG:cuda()
    user_precision = user_precision:cuda()
    user_recall = user_precision:cuda()
    user_AUC = user_AUC:cuda()
    user_map = user_map:cuda()
  end
  for i = 1, test_user_index:size(1) do
    user_NDCG[i], user_precision[i], user_recall[i], user_AUC[i], user_map[i] = NDCG_AUC(i, top_k)
  end  
  local mean_NDCG = user_NDCG:mean()
  local mean_AUC = user_AUC:mean()
  local mean_map = user_map:mean()
  print('mean NDCG:', mean_NDCG)
  print('mean AUC:', mean_AUC)
  print('mean MAP:', mean_map)
  local time_e3 = timer:time().real
  print('time of calculating NDCG: ', time_e3-time_e2)
  collectgarbage()

  local mean_NDCG = user_NDCG:mean(1):squeeze()
  local mean_precision = user_precision:mean(1):squeeze()
  local mean_recall = user_recall:mean(1):squeeze()
  for k = 1, #top_k do
    print('mean precision of top ' .. top_k[k] .. ':', mean_precision[k])
    print('mean recall of top ' .. top_k[k] .. ':', mean_recall[k])
  end
  return mean_NDCG, mean_precision, mean_recall, mean_AUC, mean_map
end

-- compute the error
local function round_v(v, opt)
  local rtn = v
  if v>5 then 
    rtn = 5
  end
  if opt.if_negative==1 then
    if v<0 then 
      rtn = 0
    end
  else
    if v<1 then 
      rtn = 1
    end
  end
  --    rtn = math.floor(rtn+0.5)
  return rtn
end

--- inference one sample
local function inference(model, user_x, item_x, true_y, opt, category, data_pair, loader)
  -- decode the model and parameters
  local category_transform = model.category_transform
  local user_attention = model.user_attention
  local user_att_weight_net = model.user_att_weight
  local item_attention = model.item_attention
  local item_att_weight_net = model.item_att_weight
  local rnn_user = model.rnn_user
  local rnn_item = model.rnn_item
  local top_net = model.top_net
  local MF_net = model.MF_net
  local bias_term = model.Bias_Term
  local criterion = model.criterion 
  local params_flat = model.params_flat

  local user_length = user_x:size(2)
  local item_length = item_x:size(2)

  -- perform the forward pass for the category transform for item time series
  local cate_item_x = nil
  if opt.if_category == 1 and category:nElement() > 0 then
    local cate_trans = category_transform.forward(category, item_x:t())
    cate_item_x = cate_trans:t()
  else
    cate_item_x = item_x
  end

    -- perform the forward pass for attention model
    local user_att_weights, user_hidden_z_value, user_hidden_summary
    local item_att_weights, item_hidden_z_value, item_hidden_summary
    if opt.user_module == 'TAGM' or opt.user_module == 'IARN' then
      user_hidden_z_value, user_hidden_summary = user_attention:forward(user_x, opt, 'test')
      item_hidden_z_value, item_hidden_summary = item_attention:forward(cate_item_x, opt, 'test')
      user_att_weights = user_att_weight_net:forward(user_hidden_z_value, item_hidden_summary)
      item_att_weights = item_att_weight_net:forward(item_hidden_z_value, user_hidden_summary)
    else
      user_att_weights = torch.ones(user_length)
      item_att_weights = torch.ones(item_length)
      if opt.gpuid >= 0 then
       user_att_weights = user_att_weights:cuda()
       item_att_weights = item_att_weights:cuda()
      end
    end

  -- perform the forward for the rnn_user
  local user_hidden = rnn_user:forward(user_x, user_att_weights, opt, 'test')
  local item_hidden = rnn_item:forward(cate_item_x, item_att_weights, opt, 'test')
  -- perform the forward for the top-net module
  local rnn_net_output = top_net:forward({user_hidden, item_hidden})
  -- perform the forward for the MF
  local one = torch.ones(1)
  if opt.gpuid >= 0 then one = one:cuda() end
  local MF_output = MF_net:forward({one, one:clone()})
  -- perform the forward for the bias term
  local bias_out = 0
  if opt.if_bias == 1 then
    bias_out = bias_term:forward(data_pair[1]+1, data_pair[2]+1)
  end
  --compute the loss
  local current_loss = nil
  if opt.if_bias == 0 then
    current_loss = criterion:forward({rnn_net_output, MF_output}, {true_y, true_y})
  else
    current_loss = criterion:forward({rnn_net_output+bias_out, MF_output}, {true_y, true_y})
  end
  if opt.if_weighted_loss == 1 then
    current_loss = current_loss * loader.class_weight[true_y:squeeze()]
  end

  local pred_rnn_v
  if opt.if_bias == 0 then
    pred_rnn_v = round_v(rnn_net_output[1], opt)
  else
    pred_rnn_v = round_v(rnn_net_output[1]+bias_out, opt)
  end

--  local MF_err = math.pow(MF_output[1]-true_y[1],2)
--  local rnn_err = math.pow(pred_rnn_v-true_y[1],2)
--  local abs_err = math.abs(pred_rnn_v-true_y[1])
--  local error = (1-opt.TAGM_weight)*MF_err + opt.TAGM_weight*rnn_err 
--  local user_weight = user_att_weights:clone()
--  local item_weight = item_att_weights:clone()
--  return current_loss, error, MF_err, rnn_err,pred_rnn_v, abs_err, user_weight, item_weight
  
  local rnn_err = math.pow(pred_rnn_v-true_y[1],2)
  local abs_err = math.abs(pred_rnn_v-true_y[1]) 
  local user_weight = user_att_weights:clone()
  local item_weight = item_att_weights:clone()
  return current_loss, rnn_err,pred_rnn_v, abs_err, user_weight, item_weight
end

--- input @data_set is a data sequence (table of Tensor) to be evaluated
local function evaluation_set_performance(opt, model, data_pairs, true_labels, if_test, set_name, loader)
  local total_loss_avg = 0
  local total_rnn_err = 0
  local total_abs_err = 0
  local accuracy = 0
  local data_size = true_labels:size(1)
  local batch_size = opt.batch_size
  local temp_idx = 1
  local cc = 1
  local all_attention_weights = {}
  local predictions = torch.zeros(data_size) 
  if opt.gpuid >= 0 then predictions = predictions:cuda() end
  local user_weights = {}
  local item_weights = {}
    
  local user_hidden, item_hidden,user_summary, item_summary, cate_item_x_set
  local savefile = string.format('%s/hidden_vec_dropout_%1.2f_negative_%d.t7', opt.current_result_dir, opt.dropout, opt.if_negative) 
  if file_exists(savefile) then
    local hidden_v = torch.load(savefile)
    if opt.user_module == 'IARN' then
      user_hidden = hidden_v.user_hidden
      user_summary = hidden_v.user_summary
      item_hidden = hidden_v.item_hidden
      item_summary = hidden_v.item_summary
      cate_item_x_set = hidden_v.cate_item_x_set
    else
      user_hidden = hidden_v.user_hidden
      item_hidden = hidden_v.item_hidden
    end
  end  

  for i = 1, data_size do 
    local user_x = loader.user_X[data_pairs[i][1]+1]
    local item_x = loader.item_X[data_pairs[i][2]+1]
    local true_y = true_labels[i]
    user_x = prepro(opt, user_x)
    item_x = prepro(opt, item_x)
    if opt.gpuid >= 0 and opt.opencl == 0 then
      true_y = true_y:float():cuda()
    end
    local category = nil
    if opt.if_category == 1 then
      category = loader.item_category[data_pairs[i][2]+1]
    end
    local temp_loss, rnn_err, pred_v, abs_err, user_att, item_att
--    if not file_exists(savefile) then
      -- method 1: from inference
      temp_loss, rnn_err, pred_v, abs_err, user_att, item_att = inference(model, user_x, item_x, true_y, opt, category, data_pairs[i], loader)
      user_weights[i] = user_att
      item_weights[i] = item_att
      if i < 3 then
        print('user:')
        print(user_att)
        print('item:')
        print(item_att)  
      end
   
    total_loss_avg = temp_loss + total_loss_avg
    total_rnn_err = total_rnn_err + rnn_err
    total_abs_err = total_abs_err + abs_err
    predictions[i] = pred_v
    if i % 1000 == 0 then
      print(i, 'finished!')
      collectgarbage()
    end
  end
  total_loss_avg = total_loss_avg / data_size
  total_rnn_err = math.sqrt(total_rnn_err / data_size)
  total_abs_err = total_abs_err / data_size
  if set_name == 'test' then
    local prediction_sort, inds = torch.sort(predictions)
    local true_sort = true_labels:index(1,inds)
    matio.save(opt.current_result_dir .. '/predictions_dropout_' .. opt.dropout .. '_negative_' .. opt.if_negative .. '_' .. set_name .. '.mat', {prediction = predictions, true_y = true_labels})
    matio.save(opt.current_result_dir .. '/prediction_sort_dropout_' .. opt.dropout .. '_negative_' .. opt.if_negative .. '_' .. set_name .. '.mat', {prediction = prediction_sort, true_y = true_sort})
  end
  
  return total_loss_avg, total_rnn_err, total_abs_err, user_weights, item_weights, predictions
end

--- evaluate the data set
function evaluate_process.evaluate_set(set_name, opt, loader, model, if_limit, if_final_evaluate)
  if_limit = if_limit or false
  if_final_evaluate = if_final_evaluate or false
  print('start to evaluate the whole ' .. set_name .. ' set...')
  local timer = torch.Timer()
  local time_s = timer:time().real
  if not if_plot then
    if_plot = false
  end
  local total_loss_avg = nil
  local mean_sqrt_total_error = nil
  local MF_err = nil
  local rnn_err = nil
  local abs_err = nil
  local accuracy = nil
  local user_weights, item_weights
  local predictions
  if set_name == 'train' then
    total_loss_avg, rnn_err, abs_err, user_weights, item_weights, predictions = evaluation_set_performance(opt, model,
      loader.train_pairs,loader.train_T, false, set_name, loader)
    if if_final_evaluate and (opt.user_module == 'IARN' or opt.user_module == 'TAGM')  then
      save_attention_weight(set_name, user_weights, item_weights, predictions, opt, loader)
    end
  elseif set_name == 'validation' then
    total_loss_avg, rnn_err, abs_err, user_weights, item_weights, predictions = evaluation_set_performance(opt, model,
      loader.validation_pairs,loader.validation_T, false, set_name, loader)
    if if_final_evaluate and (opt.user_module == 'IARN' or opt.user_module == 'TAGM')  then
      save_attention_weight(set_name, user_weights, item_weights, predictions, opt, loader)
    end
  elseif set_name == 'test' then
    if if_limit then
      total_loss_avg, rnn_err, abs_err = evaluation_set_performance(opt, model,
        loader.test_pairs:sub(1,opt.test_limit_number),loader.test_T:sub(1,opt.test_limit_number), true, set_name, loader)
    else
      total_loss_avg, rnn_err, abs_err, user_weights, item_weights, predictions = evaluation_set_performance(opt, model,
        loader.test_pairs,loader.test_T, true, set_name, loader)
      if if_final_evaluate and (opt.user_module == 'IARN' or opt.user_module == 'TAGM')  then
        save_attention_weight(set_name, user_weights, item_weights, predictions, opt, loader)
      end
    end
  else
    error('there is no such set name!')
  end 
  local time_e = timer:time().real
  print('total average loss of ' .. set_name .. ' set:', total_loss_avg)
  print('total rnn error of ' .. set_name .. ' set:', rnn_err)
  print('total abs error of ' .. set_name .. ' set:', abs_err)
  print('elapsed time for evaluating the ' .. set_name .. ' set:', time_e - time_s)

  
  return total_loss_avg, rnn_err, abs_err
end

--- load the data and the trained model from the check point and evaluate the model
function evaluate_process.evaluate_from_scratch(opt, if_train_validation)

  torch.manualSeed(opt.seed)
  ------------------- create the data loader class ----------
  local loader = data_loader.create(opt)
  local do_random_init = true

  ------------------ begin to define the whole model --------------------------
  local model = define_my_model.define_model(opt, loader, true)
--  define_my_model.load_model(opt,model, false)
  local if_plot = false
  
  local temp_file = io.open(string.format('%s/%s_results_GPU_%d_dropout_%1.2f_negative_%d.txt', 
    opt.current_result_dir, opt.opt_method, opt.gpuid, opt.dropout, opt.if_negative), "a")
    
  ----------------------hidden representation extraction--------------------------
  if opt.if_ranking == 1 then
    if opt.user_module == 'IARN' then
      evaluate_process.extract_all_hidden_representation_IARN(model, loader, opt)
    else
      evaluate_process.extract_all_hidden_representation_normal(model, loader, opt)
    end
    local NDCG, precision, recall, auc, MAP = calcuate_NDCG_AUC(opt, loader, model)

    temp_file:write(string.format('NDCG = %6.8f, AUC = %6.8f, Mean Average Precision = %6.8f\n', NDCG, auc, MAP))
    temp_file:write(string.format('precision_top_5 = %6.8f, precision_top_10 = %6.8f, precision_top_20 = %6.8f\n',
      precision[1], precision[2], precision[3]))
    temp_file:write(string.format('recall_top_5 = %6.8f, recall_top_10 = %6.8f, recall_top_20 = %6.8f\n',
      recall[1], recall[2], recall[3]))
  end
  ------------------- performance evaluation ----------
  print('evaluate the model from scratch...')
  local train_loss, train_sqrt_err, train_abs_err = nil
  local validation_loss, validation_sqrt_err, validation_abs_err = nil
  if if_train_validation then 
    train_loss, train_sqrt_err, train_abs_err = evaluate_process.evaluate_set('train', opt, loader, model, false, true)
    validation_loss, validation_sqrt_err, validation_abs_err = evaluate_process.evaluate_set('validation', opt, loader, model, false, true)
  end
  local test_loss, test_sqrt_err, test_abs_err = evaluate_process.evaluate_set('test', opt, loader, model, false, true)


  temp_file:write(string.format('results \n'))
  if if_train_validation then
    temp_file:write(string.format('train set loss = %6.8f, train sqrt error= %6.8f, train abs error= %6.8f\n', 
      train_loss, train_sqrt_err, train_abs_err ))
    temp_file:write(string.format('validation set loss = %6.8f, validation sqrt error = %6.8f, validation abs error = %6.8f\n', 
      validation_loss, validation_sqrt_err, validation_abs_err ))
  end
  temp_file:write(string.format('test set loss = %6.8f, test sqrt error = %6.8f, test abs error = %6.8f\n', 
    test_loss, test_sqrt_err, test_abs_err ))

  if if_train_validation then
    return train_sqrt_err, validation_sqrt_err, test_sqrt_err
  else
    return test_sqrt_err
  end
end

--- for different min_len
function evaluate_process.evaluate_from_scratch2(opt, if_train_validation)

  torch.manualSeed(opt.seed)
  ------------------- create the data loader class ----------
  local loader = data_loader.create(opt)
  local do_random_init = true

  ------------------ begin to define the whole model --------------------------
  local dropouts = {}
  if opt.user_module == 'IARN' then
    dropouts['Clothing'] = 0.50
    dropouts['Sports'] = 0.25
    dropouts['Electronics'] = 0.25
    dropouts['Home'] = 0.10
    dropouts['MovieLens'] = 0.00
    dropouts['Netflix'] = 0.00
  elseif opt.user_module == 'TAGM' then
    dropouts['Clothing'] = 0.50
    dropouts['Sports'] = 0.25
    dropouts['Electronics'] = 0.00
    dropouts['Home'] = 0.25
    dropouts['MovieLens'] = 0.00
    dropouts['Netflix'] = 0.00
  elseif opt.user_module == 'lstm' then
    dropouts['Clothing'] = 0.50
    dropouts['Sports'] = 0.00
    dropouts['Electronics'] = 0.00
    dropouts['Home'] = 0.00
    dropouts['MovieLens'] = 0.00
    dropouts['Netflix'] = 0.00
  else
    dropouts['Clothing'] = 0.00
    dropouts['Sports'] = 0.00
    dropouts['Electronics'] = 0.00
    dropouts['Home'] = 0.25
    dropouts['MovieLens'] = 0.00
    dropouts['Netflix'] = 0.00
  end
  
  opt.dropout = dropouts[opt.data_set]
  local model = define_my_model.define_model(opt, loader, true)
--  define_my_model.load_model(opt,model, false)
  local if_plot = false
  
  local temp_file = io.open(string.format('%s/%s_results_GPU_%d_dropout_%1.2f_negative_%d_different_min_len.txt', 
    opt.current_result_dir, opt.opt_method, opt.gpuid, opt.dropout, opt.if_negative), "a")

  local min_lens = {10, 20, 30, 50, 100}
  if opt.data_set == 'Clothing' then
    min_lens = {10, 20, 30}
  elseif opt.data_set == 'Sports' then
    min_lens = {10, 20, 30, 50}
  end
--  local min_lens = {150, 200}
  if opt.data_set == 'MovieLens' or opt.data_set == 'Netflix' then
    min_lens[#min_lens+1] = 150
    min_lens[#min_lens+1] = 200
  end
  
  for i = 1, #min_lens do
    opt.min_len = min_lens[i]
    local loader = data_loader.create(opt)
    ------------------- performance evaluation ----------
    print('evaluate the model from scratch...')
    local train_loss, train_sqrt_err, train_abs_err = nil
    local validation_loss, validation_sqrt_err, validation_abs_err = nil
    if if_train_validation then 
      train_loss, train_sqrt_err, train_abs_err = evaluate_process.evaluate_set('train', opt, loader, model, false, true)
      validation_loss, validation_sqrt_err, validation_abs_err = evaluate_process.evaluate_set('validation', opt, loader, model, false, true)
    end
    local test_loss, test_sqrt_err, test_abs_err = evaluate_process.evaluate_set('test', opt, loader, model, false, true)

    temp_file:write(string.format('model: %s, min_len = %d\n', opt.user_module, opt.min_len))
    if if_train_validation then
      temp_file:write(string.format('train set loss = %6.8f, train sqrt error= %6.8f, train abs error= %6.8f\n', 
        train_loss, train_sqrt_err, train_abs_err ))
      temp_file:write(string.format('validation set loss = %6.8f, validation sqrt error = %6.8f, validation abs error = %6.8f\n', 
        validation_loss, validation_sqrt_err, validation_abs_err ))
    end
    temp_file:write(string.format('test set loss = %6.8f, test sqrt error = %6.8f, test abs error = %6.8f\n', 
      test_loss, test_sqrt_err, test_abs_err ))
  end
  
end

--- for the gradient check
function evaluate_process.grad_check(model, user_x, item_x, true_y, opt, category, data_pair, loader)
  -- decode the model and parameters
  local start = 1
  local category_params_flat, category_grad_params_flat
  if opt.if_category == 1 and category:nElement() > 0 then
    local temp = category[1]
    category_params_flat = model.params_flat:sub(start+model.category_param_size/1441*temp, start+model.category_param_size/1441*(temp+1)-1)
    category_grad_params_flat = model.grad_params_flat:sub(start+model.category_param_size/1441*temp, start+model.category_param_size/1441*(temp+1)-1)
    start = model.category_param_size+1
  end
  local user_att_flat = model.params_flat:sub(start, start+model.user_attention.params_size)
  local user_att_grad_flat = model.grad_params_flat:sub(start, start+model.user_attention.params_size)
  start = start+model.user_attention.params_size
    local user_att_weight_flat = model.params_flat:sub(start, start+model.user_att_weight.params_size)
  local user_att_weight_grad_flat = model.grad_params_flat:sub(start, start+model.user_att_weight.params_size)
  start = start-model.user_attention.params_size
  local user_params_flat = model.params_flat:sub(start, start+model.user_params_size)
  local user_grad_flat = model.grad_params_flat:sub(start, start+model.user_params_size)
  start = start + model.user_params_size
  local item_params_flat = model.params_flat:sub(start, start+model.item_params_size)
  local item_grad_flat = model.grad_params_flat:sub(start, start+model.item_params_size)
  
  if opt.if_bias == 1 then
    local user_ind = data_pair[1]+1+model.params_size-model.bias_term_params_size
    local item_ind = data_pair[2]+1+model.Bias_Term.user_bias_size+model.params_size-model.bias_term_params_size
    local bias_term_params = model.params_flat:sub(user_ind, user_ind)
    local bias_term_grad_params = model.grad_params_flat:sub(user_ind, user_ind)
  end
  local total_params = model.params_size
  local function calculate_loss()
    local current_loss = inference(model, user_x, item_x, true_y, opt, category, data_pair, loader)
    return current_loss
  end  

  local function gradient_compare(params, grad_params)
    local check_number = math.min(200, params:nElement())
    local loss_minus_delta, loss_add_delta, grad_def
    if opt.gpuid >= 0 then
      loss_minus_delta = torch.CudaTensor(check_number)
      loss_add_delta = torch.CudaTensor(check_number)
      grad_def = torch.CudaTensor(check_number)
    else
      loss_minus_delta = torch.DoubleTensor(check_number)
      loss_add_delta = torch.DoubleTensor(check_number)
      grad_def = torch.DoubleTensor(check_number)    
    end
    local params_backup = params:clone()
    local rand_ind = torch.randperm(params:nElement())
    rand_ind = rand_ind:sub(1, check_number)
    for k = 3, 6 do
      local delta = 1 / torch.pow(1e1, k)
      print('delta:', delta)
      for i = 1, check_number do
        local ind = rand_ind[i]
        params[ind] = params[ind] - delta
        loss_minus_delta[i] = calculate_loss() 
        params[ind] = params[ind] + 2*delta
        loss_add_delta[i] = calculate_loss()
        local gradt = (loss_add_delta[i] - loss_minus_delta[i]) / (2*delta)
        grad_def[i] = gradt
        params[ind] = params[ind] - delta -- retore the parameters
        if i % 100 ==0 then
          print(i, 'processed!')
        end
      end
      params:copy(params_backup) -- retore the parameters
      local grad_model = grad_params:index(1, rand_ind:long())
      local if_print = true
      local threshold = 1e-4
      local inaccuracy_num = 0
      local reversed_direction = 0
      assert(grad_def:nElement()==grad_model:nElement())
      local relative_diff = torch.zeros(grad_def:nElement())
      relative_diff = torch.abs(grad_def - grad_model)
      relative_diff:cdiv(torch.cmax(torch.abs(grad_def), torch.abs(grad_model)))
      for i = 1, grad_def:nElement() do
        if if_print then
          print(string.format('index: %4d, rand_index: %4d, relative_diff: %6.5f,  gradient_def: %6.25f,  grad_model: %6.25f',
            i, rand_ind[i], relative_diff[i], grad_def[i], grad_model[i]))
        end
        if relative_diff[i] > threshold then
          if math.max(math.abs(grad_def[i]), math.abs(grad_model[i])) > 1e-8 then
            inaccuracy_num = inaccuracy_num + 1
          end   
        end
      end
      for i = 1, grad_def:nElement() do
        if grad_def[i] * grad_model[i] < 0 then
          if if_print then
            print(string.format('index: %4d, relative_diff: %6.5f,  gradient_def: %6.10f,  grad_params: %6.10f',
              i, relative_diff[i], grad_def[i], grad_model[i]))
          end
          reversed_direction = reversed_direction + 1
        end
      end

      print('there are', inaccuracy_num, 'inaccuracy gradients.')
      print('there are', reversed_direction, 'reversed directions.')
    end
  end


---- check the bias term
--gradient_compare(bias_term_params, bias_term_grad_params)
  -- check the category_transform
--  gradient_compare(category_params_flat, category_grad_params_flat)
--  --     check rnn params  
     gradient_compare(user_att_flat, user_att_grad_flat)
--        gradient_compare(user_att_weight_flat, user_att_weight_grad_flat)
--        gradient_compare(item_params_flat, item_grad_flat)  
  --  --  --   check top_net params
  --          gradient_compare(top_net_params_flat, top_net_grad_flat)


end

return evaluate_process

