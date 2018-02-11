local path = require 'pl.path'
local table_operation = require 'util.table_operation'
require 'util.misc'

local data_loader = {}
data_loader.__index = data_loader


function data_loader.create(opt)
  local self = {}
  setmetatable(self, data_loader)
  self:load_data_by_index_t7(opt)

  self.batch_ix = 1
  self.rand_order = torch.randperm(self.nTrain)
  print('data load done. ')
  print('before collectgarbage: ', collectgarbage("count"))
  collectgarbage()
  print('after collectgarbage: ', collectgarbage("count"))
  --  os.exit()
  return self
end

-- Return the tensor of index of the data with batch size 
function data_loader:get_next_train_batch(batch_size)
  self.previous_batch_ix = self.batch_ix
  local set_size = self.nTrain;
  local rtn_index = torch.zeros(batch_size)
  for i = 1, batch_size do
    local temp_ind = i + self.batch_ix - 1
    if temp_ind > set_size then -- cycle around to beginning
      temp_ind = temp_ind - set_size
    end
    rtn_index[i] = self.rand_order[temp_ind]
  end
  self.batch_ix = self.batch_ix + batch_size;
  -- cycle around to beginning
  if self.batch_ix >set_size then
    self.batch_ix = 1
    -- randomize the order of the training set
    self.rand_order = torch.randperm(self.nTrain)
  end
  return rtn_index;
end

function data_loader:load_data_by_index_t7(opt)
  print('loading ', opt.data_set, ' data: ')
  local item_category_dir = nil
  local test_rating_dir = nil
  local train_rating_dir = nil
  local item_time_series_dir = nil
  local user_time_series_dir =  nil
  --  if opt.data_set == 'Clothing' then
  item_category_dir = path.join(opt.data_dir, opt.data_set, 't7_format/item_category.t7')
  test_rating_dir = path.join(opt.data_dir, opt.data_set, 't7_format/test_rating.t7')
  train_rating_dir = path.join(opt.data_dir, opt.data_set, 't7_format/training_rating.t7')
  if opt.if_new_data == 1 then
    item_time_series_dir = path.join(opt.data_dir, opt.data_set, 't7_format/training_item_time_series_minlength3.t7')
    user_time_series_dir = path.join(opt.data_dir, opt.data_set, 't7_format/training_user_time_series_minlength3.t7')
  else
    item_time_series_dir = path.join(opt.data_dir, opt.data_set, 't7_format/training_item_time_series.t7')
    user_time_series_dir = path.join(opt.data_dir, opt.data_set, 't7_format/training_user_time_series.t7')
  end

  local item_category = nil
  if file_exists(item_category_dir) then
    item_category = torch.load(item_category_dir)
    item_category = item_category.item_category
  end
  local train_pair_rating = torch.load(train_rating_dir)
  train_pair_rating = train_pair_rating.training_rating
  local test_pair_rating = torch.load(test_rating_dir)
  test_pair_rating = test_pair_rating.test_rating
  
  local remove_user_ind = {}
  local remove_item_ind = {}
  local valid_user_ind = {}
  local valid_item_ind = {}
  local avg_user_len = 0
  local avg_item_len = 0
  local item_time_series, user_time_series
  if opt.if_new_data == 0 then
    item_time_series = torch.load(item_time_series_dir)
    item_time_series = item_time_series.item_time_series
    user_time_series = torch.load(user_time_series_dir)
    user_time_series = user_time_series.user_time_series
  else
    local item_time_seriesd = torch.load(item_time_series_dir)
    item_time_series = item_time_seriesd.item_time_series
    print('current item length:', #item_time_series)
    local item_ind = item_time_seriesd.items_in_time_series
    item_ind = item_ind:squeeze()
    local max_i = math.max(train_pair_rating:select(2,2):max(), test_pair_rating:select(2,2):max()) + 1
    print('max item length:',max_i)
    local ind = 1
    local new_item_time_series = {}
    local tem = torch.ones(1, 1) 
    for i = 1, max_i do
      new_item_time_series[i] = tem 
    end
    local rec = torch.zeros(max_i)
    for i = 1, item_ind:nElement() do
      new_item_time_series[item_ind[i]+1] = item_time_series[i]
      valid_item_ind[#valid_item_ind+1] = item_ind[i]+1
      rec[item_ind[i]+1] = 1
      avg_item_len = avg_item_len + item_time_series[i]:size(1)
    end
    avg_item_len = avg_item_len / item_ind:nElement()
    for i = 1, rec:size(1) do
      if rec[i] == 0 then
        remove_item_ind[#remove_item_ind+1] = i
      end
    end
    item_time_series = new_item_time_series
    
    local user_time_seriesd = torch.load(user_time_series_dir)
    user_time_series = user_time_seriesd.user_time_series
    print('current user length:', #user_time_series)
    local user_ind = user_time_seriesd.users_in_time_series
    user_ind = user_ind:squeeze()
    local max_i = user_ind:max()+1
    local max_i = math.max(train_pair_rating:select(2,1):max(), test_pair_rating:select(2,1):max()) + 1
    print('max user length:',max_i)
    local ind = 1
    local new_user_time_series = {}
    local tem = torch.ones(1, 1) 
    for i = 1, max_i do
      new_user_time_series[i] = tem 
    end
    local rec = torch.zeros(max_i)
    for i = 1, user_ind:nElement() do
      new_user_time_series[user_ind[i]+1] = user_time_series[i]
      valid_user_ind[#valid_user_ind+1] = user_ind[i]+1
      rec[user_ind[i]+1] = 1
      avg_user_len = avg_user_len + user_time_series[i]:size(1)
    end
    avg_user_len = avg_user_len / user_ind:nElement()
    for i = 1, rec:size(1) do
      if rec[i] == 0 then
        remove_user_ind[#remove_user_ind+1] = i
      end
    end
    user_time_series = new_user_time_series
  end
  print('avg user len:', avg_user_len)
  print('avg item len:', avg_item_len)
  
  -- process the item categories info
  -- note that the index of item and category should be from 0, hence should minus 1 when used
  if opt.if_category == 1 then
    self.max_category = item_category:select(2,2):max()
    local item_category_tidy = {}
    local ind = 0
    local temp = {}
    for i = 1, item_category:size(1) do
      if item_category[i][1] == ind then
        temp[#temp+1] = item_category[i][2]
      else
        temp = torch.Tensor(temp):sort()
        temp = temp:sort()
        item_category_tidy[#item_category_tidy+1] = temp
        temp = {}
        temp[#temp+1] = item_category[i][2]
        ind = ind + 1
      end
    end
    item_category_tidy[#item_category_tidy+1] = torch.Tensor(temp):sort() 
    self.item_category = item_category_tidy
  end
  --  for i = #item_category_tidy, #item_category_tidy-10, -1 do
  --    print(item_category_tidy[i])
  --  end 
  --  os.exit() 

  -- remove the time series whose length < min_len
  local user_max_len = 0
  local item_max_len = 0

  if opt.if_new_data == 0 or opt.min_len > 3 then
    remove_user_ind = {}
    remove_item_ind = {}
    valid_user_ind = {}
    valid_item_ind = {}
  end
  local avg_user_len = 0
  local avg_item_len = 0
  for i = 1, #user_time_series do
    if user_time_series[i]:size(1) >= opt.min_len then
      user_max_len = math.max(user_time_series[i]:size(1), user_max_len)
      if opt.if_new_data == 0 or opt.min_len > 3 then
        valid_user_ind[#valid_user_ind+1] = i
      end
    else
      if opt.if_new_data == 0 or opt.min_len > 3 then
        remove_user_ind[#remove_user_ind+1] = i
      end
    end
    user_time_series[i] = user_time_series[i]:t()
  end
  for i = 1, #item_time_series do
    if item_time_series[i]:size(1) >= opt.min_len then
      item_max_len = math.max(item_time_series[i]:size(1), item_max_len)
      if opt.if_new_data == 0 or opt.min_len > 3 then
        valid_item_ind[#valid_item_ind+1] = i
      end
    else
      if opt.if_new_data == 0 or opt.min_len > 3 then
        remove_item_ind[#remove_item_ind+1] = i
      end
    end
    item_time_series[i] = item_time_series[i]:t()
  end

  self.user_X = user_time_series
  self.item_X = item_time_series
  for i = 1, #self.user_X do
    local xx = self.user_X[i]
    if xx == nil then
      print(i)
      error('element of self_user_X is nil!')
    end
  end
  for i = 1, #self.item_X do
    local xx = self.item_X[i]
    if xx:size():size(1) ~= 2 then 
      print(xx)
      error('xx:size():nElement() ~= 2') 
    end
    if xx == nil then
      print(i)
      error('element of self_item_X is nil!')
    end
  end
  self.user_max_len = user_max_len
  self.item_max_len = item_max_len
  self.feature_dim = user_time_series[1]:size(1)
  remove_user_ind = torch.Tensor(remove_user_ind)
  remove_item_ind = torch.Tensor(remove_item_ind)
  valid_user_ind = torch.Tensor(valid_user_ind)
  valid_item_ind = torch.Tensor(valid_item_ind)
--  print(remove_user_ind:size(1)+valid_user_ind:size(1))
--  print(remove_item_ind:size(1)+valid_item_ind:size(1))
  if opt.min_len > 1 then
    print('valid user size:', (valid_user_ind:size(1)) .. ' / ' .. #user_time_series)
    print('valid item size:', (valid_item_ind:size(1)) .. ' / ' .. #item_time_series)
  end
  local valid_ind = {}
  for i = 1, train_pair_rating:size(1) do
    if torch.sum(remove_user_ind:eq(train_pair_rating[i][1]+1)) == 0 and
      torch.sum(remove_item_ind:eq(train_pair_rating[i][2]+1)) == 0 then
      valid_ind[#valid_ind+1] = i
--      local ul = self.user_X[train_pair_rating[i][1]+1]:size(2)
--      local il = self.item_X[train_pair_rating[i][2]+1]:size(2)
----      print(ul, il)
--      if ul<opt.min_len or il < opt.min_len then error('ul<opt.min_len or il<opt.min_len') end
    end
  end
  self.remove_user_ind = remove_user_ind
  self.remove_item_ind = remove_item_ind
  self.valid_user_ind = valid_user_ind
  self.valid_item_ind = valid_item_ind
  valid_ind = torch.LongTensor(valid_ind)
  train_pair_rating = train_pair_rating:index(1, valid_ind)                                                       
  local train_pairs = train_pair_rating:index(2, torch.LongTensor{1,2})
  local train_T = train_pair_rating:index(2, torch.LongTensor{3})
  
  self.class_size = train_T:max()
  local each_class_size = torch.Tensor(self.class_size)
  local each_class_weight = torch.Tensor(self.class_size)
  for i = 1, self.class_size do
    local n = train_T:eq(i)
    each_class_size[i] = torch.sum(n)
    print('grade ' .. i .. ':', torch.sum(n))
    each_class_weight[i] = train_T:nElement() / torch.sum(n) 
  end 
  self.class_weight = each_class_weight
  print(self.class_weight)
  
  -- add the negative sample (label = 0) pairs to balance the data
  if opt.if_negative == 1 then
    local data_f = opt.data_dir .. '/' .. opt.data_set .. '/train_data_with0.t7'
    if file_exists(data_f) then
      local dd = torch.load(data_f)
      train_pairs = dd.train_pairs
      train_T = dd.train_T
    else
      local map_ind = torch.ByteTensor(#self.user_X, #self.item_X):fill(0)
      for m = 1, train_pairs:size(1) do
        local u = train_pairs[m][1]+1
        local v = train_pairs[m][2]+1
        map_ind[u][v] = 1
      end
      local max_size = each_class_size:max() / 2
      local neg_pair = torch.DoubleTensor(max_size, 2)
      local neg_T = torch.zeros(max_size, 1)
      local user_randp = torch.randperm(self.valid_user_ind:nElement())
      local item_randp = torch.randperm(self.valid_item_ind:nElement())
      local function if_valid(u, p)
        if map_ind[u+1][p+1] == 1 then
          return false
        else
          return true
        end 
      end
      local indss = 0
      local indu = 1
      local indi = 1
      while 1 do
        local user_i = self.valid_user_ind[user_randp[indu]]-1
        local item_i = self.valid_item_ind[item_randp[indi]]-1
        if if_valid(user_i, item_i) then
          neg_pair[indss+1] = torch.Tensor({user_i, item_i}) 
          indss = indss + 1
          if indss % 100 == 0 then
            print(indss)
          end
          if indss == neg_pair:size(1) then break end
        end
        indu = indu + 1
        if indu > user_randp:size(1) then 
          user_randp = torch.randperm(self.valid_user_ind:nElement())
          indu = 1 
        end
        indi = indi + 1
        if indi > item_randp:size(1) then
          item_randp = torch.randperm(self.valid_item_ind:nElement())
          indi = 1 
        end
      end
      train_pairs = torch.cat(train_pairs, neg_pair, 1)
      train_T = torch.cat(train_T, neg_T, 1)
      print('size of negative pairs: ', neg_pair:size())
      print('size of negative pairs: ', neg_T:size())

      local randpp = torch.randperm(train_T:size(1))
      train_pairs = train_pairs:index(1, randpp:long())
      train_T = train_T:index(1, randpp:long())

      local train_0 = {
        train_pairs = train_pairs,
        train_T = train_T
      }
      torch.save(data_f, train_0)
    end
    self.class_size = train_T:max()
    for i = 0, self.class_size do
      local n = train_T:eq(i)
      print('grade ' .. i .. ':', torch.sum(n))
    end
  end
   
  local randp = torch.randperm(train_pairs:size(1))
  local validation_s = math.min(math.floor(opt.validation_size * train_pairs:size(1)), opt.validation_max_size)
  self.validation_pairs = train_pairs:index(1, randp:sub(1, validation_s):long())
  self.train_pairs = train_pairs:index(1, randp:sub(1+validation_s, -1):long())
  self.validation_T = train_T:index(1, randp:sub(1, validation_s):long())
  self.train_T = train_T:index(1, randp:sub(1+validation_s, -1):long())
  self.user_max_index = self.train_pairs:select(2,1):max()
  self.item_max_index = self.train_pairs:select(2,2):max()
  print('max user index:', self.user_max_index)
  print('max item index:', self.item_max_index)
  -- get the mean value of the rating in the training set
  local user_mean_rate = torch.Tensor(#self.user_X)
  local item_mean_rate = torch.Tensor(#self.item_X)
  for i = 1, user_mean_rate:size(1) do
    local indx = self.train_pairs:select(2,1):eq(i-1)
    if torch.sum(indx)>0 then
      user_mean_rate[i] = self.train_T[indx]:mean()
    else
      user_mean_rate[i] = 1
    end
  end
  for i = 1, item_mean_rate:size(1) do
    local indx = self.train_pairs:select(2,2):eq(i-1)
    if torch.sum(indx)>0 then
      item_mean_rate[i] = self.train_T[indx]:mean()
    else
      item_mean_rate[i] = 1
    end
  end
  self.user_mean_rate = user_mean_rate
  self.item_mean_rate = item_mean_rate
  valid_ind = {}
  for i = 1, test_pair_rating:size(1) do
    if torch.sum(remove_user_ind:eq(test_pair_rating[i][1]+1)) == 0 and
      torch.sum(remove_item_ind:eq(test_pair_rating[i][2]+1)) == 0 then
      valid_ind[#valid_ind+1] = i
    end
  end
  valid_ind = torch.LongTensor(valid_ind)
  test_pair_rating = test_pair_rating:index(1, valid_ind)
  self.test_pairs = test_pair_rating:index(2, torch.LongTensor{1,2})
  self.test_T = test_pair_rating:index(2, torch.LongTensor{3})
  local randpt = torch.randperm(self.test_T:nElement())
  self.test_pairs = self.test_pairs:index(1, randpt:long())
  self.test_T = self.test_T:index(1, randpt:long())

  self.nTrain = self.train_T:size(1)
  self.nValidation = self.validation_T:size(1)
  self.nTest = self.test_T:size(1)
  local data_size = self.train_T:nElement()+self.validation_T:nElement()+self.test_T:nElement()
 

  local mean_train_T = self.train_T:mean()
  local mean_validation_T = self.validation_T:mean()
  local mean_test_T = self.test_T:mean()
  print('mean train T: ', mean_train_T)
  print('mean validation T: ', mean_validation_T)
  print('mean test T: ', mean_test_T)
  print('grade number: ', self.class_size)
  print('training size: ', self.train_T:size(1))
  print('validation size: ', self.validation_T:size(1))
  print('test size: ', self.test_T:size(1))
  print('user max length: ', user_max_len)
  print('item max length: ', item_max_len)
  print('feature dim: ', self.feature_dim)
  print('max category: ', self.max_category)
  print('#user_X:', #self.user_X)
  print('#item_X:', #self.item_X)
end

return data_loader
