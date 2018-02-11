

require 'nn'
require 'nngraph'
local path = require 'pl.path'

-- local library
local RNN = require 'model.my_RNN'
local LSTM = require 'model.my_LSTM'
local attention = require 'model.attention'
local attention_weight = require 'model.attention_weight'
local model_utils = require 'util.model_utils'
local data_loader = require 'util.data_loader'
require 'util.misc'
local Top_NN = require 'model.Top_NN_Classifier'
local Recurrent_Nets = require 'model.Recurrent_Nets'
local Top_RNN = require 'model.Top_RNN_Classifier'
local TAGM = require 'model.TAGM'
local my_criterion = require 'model.my_criterion' 
local Top_Net = require 'model.Top_Net'
local MF = require 'model.MF'
require 'util.OneHot'
local category_transform = require 'model.category_transformation'
local sum_category_transf = require 'model.category_weighted_sum'
local Bias_Term = require 'model.bias_term'

local my_model = {}

--- save the current trained best model
-- for the continution of training or for the test
function my_model.save_model(opt, model)
  local savefile = my_model.savefile
  print('saving checkpoint to ' .. savefile)
  opt.savefile = savefile
  -- to save the space, we only save the parameter values in the model
  local checkpoint = {
    params_flat = model.params_flat,
    learning_rate = opt.learning_rate
  }
  torch.save(savefile, checkpoint)
end

function my_model.load_model(opt, model, if_learning_rate)
  print(my_model.savefile)
  local savefile = my_model.savefile
  local checkpoint = torch.load(savefile)
--  print(checkpoint.params_flat:size())
  model.params_flat:copy(checkpoint.params_flat)
  if if_learning_rate then
    opt.learning_rate = checkpoint.learning_rate 
  end
  checkpoint = nil
  collectgarbage()
  return model, opt
end

function my_model.define_model(opt, loader, if_evaluate_from_scratch)
  my_model.savefile = string.format('%s/best_trained_model_GPU_%d_dropout_%1.2f_negative_%d.t7', 
    opt.current_result_dir, opt.gpuid, opt.dropout, opt.if_negative) 
  local savefile = my_model.savefile
  if_evaluate_from_scratch = if_evaluate_from_scratch or false
  local model = {}
  local models_list = {}

  ------------------- category transform ---------------------
  if opt.if_category == 1 then
    if opt.category_way == 'mul' then
      model.category_transform = category_transform
      model.category_transform.free_memory()
      model.category_transform.model(loader)
    elseif opt.category_way == 'sum' then
      model.category_transform = sum_category_transf
      model.category_transform.free_memory()
      model.category_transform.model(loader, opt)
    else
      error('no such category option!')
    end
    for i = 1, #(model.category_transform.net) do
      models_list[#models_list+1] = model.category_transform.net[i] 
    end
    model.category_param_size = model.category_transform.param_size
    print('parameter number of category_transform is: ', model.category_param_size) 
  end

  ------------------ for user module --------------------------
  local user_attention = nil
  local rnn_user = nil
  local user_att_weight = nil
  print('creating user module with ' .. opt.user_module)
  if opt.user_module == 'TAGM' or opt.user_module == 'IARN' then
    user_attention = attention.new(loader, opt)    
    if opt.user_module == 'TAGM' then
      user_att_weight = attention_weight.new(opt, false)
    else
      user_att_weight = attention_weight.new(opt, true)
    end
    rnn_user = TAGM.new(loader, opt)
    models_list[#models_list+1] = user_attention.rnn
    models_list[#models_list+1] = user_attention.birnn
    models_list[#models_list+1] = user_att_weight.weight_net
    models_list[#models_list+1] = rnn_user.pre_m
    models_list[#models_list+1] = rnn_user.rnn
    if opt.if_bidirection == 1 then
      models_list[#models_list+1] = rnn_user.birnn
    end
    model.user_params_size = user_attention.params_size + user_att_weight.params_size + rnn_user.params_size
  elseif opt.user_module == 'lstm' or opt.user_module == 'rnn' or opt.user_module == 'gru' then
    rnn_user = Recurrent_Nets.new(loader, opt, opt.user_module)
    models_list[#models_list+1] = rnn_user.mul_net
    models_list[#models_list+1] = rnn_user.rnn
    if opt.if_bidirection == 1 then
      models_list[#models_list+1] = rnn_user.birnn
    end
    model.user_params_size = rnn_user.params_size
  else
    error('no such module!')
  end
  model.user_attention = user_attention
  model.user_att_weight = user_att_weight
  model.rnn_user = rnn_user
  print('number of parameters in the user module: ', model.user_params_size)
  
  ----------------- for item module ------------------------- 
  local item_attention = nil
  local rnn_item = nil
  local item_att_weight = nil
  print('creating item module with ' .. opt.item_module)
  if opt.item_module == 'TAGM' or opt.item_module == 'IARN' then
    item_attention = attention.new(loader, opt)
    if opt.item_module == 'TAGM' then
      item_att_weight = attention_weight.new(opt, false)
    else
      item_att_weight = attention_weight.new(opt, true)
    end
    rnn_item = TAGM.new(loader, opt)
    models_list[#models_list+1] = item_attention.rnn
    models_list[#models_list+1] = item_attention.birnn
    models_list[#models_list+1] = item_att_weight.weight_net
    models_list[#models_list+1] = rnn_item.pre_m
    models_list[#models_list+1] = rnn_item.rnn
    if opt.if_bidirection == 1 then
      models_list[#models_list+1] = rnn_item.birnn
    end
    model.item_params_size = item_attention.params_size + item_att_weight.params_size + rnn_item.params_size
  elseif opt.item_module == 'lstm' or opt.item_module == 'rnn' or opt.item_module == 'gru' then
    rnn_item = Recurrent_Nets.new(loader, opt, opt.item_module)
    models_list[#models_list+1] = rnn_item.mul_net
    models_list[#models_list+1] = rnn_item.rnn
    if opt.if_bidirection == 1 then
      models_list[#models_list+1] = rnn_item.birnn
    end
    model.item_params_size = rnn_item.params_size
  else
    error('no such module!')
  end
  model.item_attention = item_attention
  model.item_att_weight = item_att_weight
  model.rnn_item = rnn_item
  print('number of parameters in the item module: ', model.item_params_size)
  
  ----------- define the top net-----------
  local top_net = Top_Net.dot_product_net(opt.rnn_size, opt)
  model.top_net = top_net
  local top_net_params_flat, _ = top_net:getParameters()
  model.top_net_params_size = top_net_params_flat:nElement()
  print('number of parameters in the top net: ', model.top_net_params_size)
  models_list[#models_list+1] = top_net
  
  ------------define the MF-------------
  model.MF_net = MF.net(opt.rnn_size) 
  local MF_params_flat, _ = model.MF_net:getParameters()
  model.MF_params_size = MF_params_flat:nElement()
  print('number of parameters in the MF net: ', model.MF_params_size)
  models_list[#models_list+1] = model.MF_net
  
  -------------define bias term---------------
  if opt.if_bias == 1 then
    model.Bias_Term = Bias_Term
    model.Bias_Term:model(loader)
    models_list[#models_list+1] = model.Bias_Term
    model.bias_term_params_size = model.Bias_Term.params_size
    print('number of parameters in the Bias_Term net: ', model.bias_term_params_size)
  end

  for k = 1, #models_list do
    if opt.gpuid >= 0 then models_list[k]:cuda() end
  end

  --------- define the criterion (loss function) ---------------
  model.criterion = my_criterion.criterion(opt.TAGM_weight)
  -- ship the model to the GPU if desired
  if opt.gpuid >= 0 then model.criterion:cuda() end 

  --------------- get the flat parameters from all modules ---------------
  local params_flat, grad_params_flat = model_utils.combine_all_parameters(unpack(models_list))
  print('number of parameters in the whole model: ' .. params_flat:nElement())
  -- clone the rnn and birnn
  if opt.user_module == 'TAGM' or  opt.user_module == 'IARN' then
    print('cloining in the user attention module')
    user_attention:clone_model(loader.user_max_len)
  end
  print('cloining in the top user rnn module')
  rnn_user:clone_model(loader.user_max_len, opt) 
  if opt.item_module == 'TAGM' or opt.item_module == 'IARN' then
    print('cloining in the item attention module')
    item_attention:clone_model(loader.item_max_len)
  end
  print('cloining in the top item rnn module')
  rnn_item:clone_model(loader.item_max_len, opt)
  model.params_flat = params_flat
  model.grad_params_flat = grad_params_flat
  model.params_size = params_flat:nElement()  

  if opt.if_init_from_check_point or if_evaluate_from_scratch then
    if path.exists(savefile) then
      print('Init the model from the check point saved before...\n')
      model, opt = my_model.load_model(opt, model, true)
    elseif if_evaluate_from_scratch then
      error('error: there is no trained model saved before in such experimental setup.')
    else    
      rnn_user:init_params(opt)
      rnn_item:init_params(opt)  
    end
  else    
    rnn_user:init_params(opt)
    rnn_item:init_params(opt)  
  end  

  -- pre-allocate the memory for the temporary variable used in the training phase
  local params_grad_all_batches = torch.zeros(grad_params_flat:nElement())
  if opt.gpuid >= 0 then
    params_grad_all_batches = params_grad_all_batches:float():cuda()
  end
  model.params_grad_all_batches = params_grad_all_batches
  collectgarbage()
  return model, opt
end

return my_model
