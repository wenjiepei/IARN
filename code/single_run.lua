
require 'torch'
require 'torch'
require 'nn'
require 'nngraph'
require 'lfs'
local path = require 'pl.path'

local train_process = require 'train_process'
local evaluate_process = require 'evaluate_process'

local single_run = {}

function single_run.run_single_experiment(opt)

  print(string.format('user_module = %s, item_module = %s \n', opt.user_module, opt.item_module ))
  print(string.format('rnn_size = %d\n', opt.rnn_size ))
  print(string.format('if_category = %d \n', opt.if_category ))
  print(string.format('if_negative = %d \n', opt.if_negative ))
  print(string.format('if_weight_dot_product = %d\n', opt.if_weighted_dot_product))
  print(string.format('dropout = %1.2f\n', opt.dropout))
  if opt.gpuid < 0 then                                                                                                   
    print('Perform calculation by CPU using the optimization method: ' .. opt.opt_method)                                         
  else
    print('Perform calculation by GPU with OpenCL using the optimization method: ' .. opt.opt_method)
  end

  -- decrease the learning rate proportionally
  opt.learning_rate = opt.original_learning_rate
  local factor = 32 / opt.rnn_size
  opt.learning_rate = opt.learning_rate * factor 

  --------- initialize cunn/cutorch for training on the GPU and fall back to CPU gracefully ------
  ------------------------------------------------------------------------------------------------
  if opt.gpuid >= 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then print('package cunn not found!') end
    if not ok2 then print('package cutorch not found!') end
    if ok and ok2 then
      print('using CUDA on GPU ' .. opt.gpuid .. '...')
      cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
      cutorch.manualSeed(opt.seed)
    else
      print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
      print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
      print('Falling back on CPU mode')
      opt.gpuid = -1 -- overwrite user setting
    end
  end

  -- make sure that output directory exists
  if not path.exists(opt.result_dir) then lfs.mkdir(opt.result_dir) end
  local current_result_dir = path.join(opt.result_dir, opt.data_set)
  if not path.exists(current_result_dir) then lfs.mkdir(current_result_dir) end
  current_result_dir = path.join(current_result_dir, 'min-len_' .. opt.min_len)
  if not path.exists(current_result_dir) then lfs.mkdir(current_result_dir) end
  current_result_dir = path.join(current_result_dir, 'if-category_' .. opt.if_category .. '__if-bias_' .. opt.if_bias .. '__user_' .. opt.user_module .. '__item_' .. opt.item_module)
  if not path.exists(current_result_dir) then lfs.mkdir(current_result_dir) end
  current_result_dir = path.join(current_result_dir, 'att_' .. opt.att_layers .. '_' .. opt.att_size .. '_sig-w_' .. opt.att_sig_w .. 
    '__rnn_' .. opt.rnn_layers .. '_' .. opt.rnn_size .. '_if-bidirection_' .. opt.if_bidirection ..
    '__dot-prod-weighted_' .. opt.if_weighted_dot_product .. '_if-PReLU_' .. opt.if_PReLU)
  if not path.exists(current_result_dir) then lfs.mkdir(current_result_dir) end
  if opt.if_category == 1 and opt.category_way == 'sum' then 
    current_result_dir = path.join(current_result_dir, 'category_weight_' .. opt.category_weight)
    if not path.exists(current_result_dir) then lfs.mkdir(current_result_dir) end
  end
  opt.current_result_dir = current_result_dir  

  local results_record = path.join(opt.result_dir, opt.data_set)
  local temp_file = io.open(string.format('%s/result_summary.txt', results_record), "a") 
  temp_file:write(string.format('user_module = %s, item_module = %s \n', opt.user_module, opt.item_module ))
  temp_file:write(string.format('rnn_size = %d\n', opt.rnn_size ))
  temp_file:write(string.format('if_category = %d \n', opt.if_category ))
  temp_file:write(string.format('if_negative = %d \n', opt.if_negative ))
  temp_file:write(string.format('if_weight_dot_product = %d\n', opt.if_weighted_dot_product))
  temp_file:write(string.format('dropout = %1.2f\n', opt.dropout))
  local if_train = opt.if_train_validation
  local train_err, validation_err, test_err = nil
  local function evaluate()
    if if_train then
      train_err, validation_err, test_err = evaluate_process.evaluate_from_scratch(opt, if_train) 
      temp_file:write(string.format('results \n'))
      temp_file:write(string.format('train err = %6.8f, validation err = %6.8f, test err = %6.8f\n', 
        train_err, validation_err, test_err ))
    else
      test_err = evaluate_process.evaluate_from_scratch(opt, if_train)
      temp_file:write(string.format('test err = %6.8f\n', test_err ))
    end
  end
  if opt.if_direct_test_from_scratch then
    evaluate()
  else
    -- begin to train the model
    print('Begin to train the model...')
    train_process.train(opt)
    print("Training Done!")
    torch.manualSeed(opt.seed)
    evaluate()
  end
  temp_file:write(string.format('\n\n\n'))
  temp_file:close()
  
  return test_err
end

return single_run

