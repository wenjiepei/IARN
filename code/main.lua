--[[

This is the implementation of the Interacting Attention-gated Recurrent Networks (IARNs)

Acknowledgement: this code is based on the code 'https://github.com/karpathy/char-rnn', deveploped by Andrej karpathy.

Copyright (c) 2017 Wenjie Pei
Delft University of Technology 

]]--

require 'torch'
require 'nn'
require 'nngraph'
require 'lfs'
local path = require 'pl.path'

local train_process = require 'train_process'
local evaluate_process = require 'evaluate_process'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Train an attention-recsys model')
cmd:text()
cmd:text('Options')
--- data
cmd:option('-data_dir','../../data/','data directory.') 
cmd:option('-data_set', 'Home')
cmd:option('-result_dir','result','result directory.')
cmd:option('-test_fold_index', 1, 'the test fold index')
cmd:option('-validation_size', 0.1, 'the size of validation set') 
cmd:option('-validation_max_size', 8000)
cmd:option('-min_len', 3)
cmd:option('-if_negative', 0)
cmd:option('-if_new_data', 1)

--- model params
cmd:option('-model_type', 'IARN', 'option: TAGM, rnn, lstm, IARN')
cmd:option('-user_module', 'IARN', 'option: TAGM, rnn, lstm, IARN (interactive attention gated recurrent network)')
cmd:option('-item_module', 'IARN', 'option: TAGM, rnn, lstm, IARN')
cmd:option('-rnn_size', 64, 'size of rnn internal state (for top rnn module)')
cmd:option('-rnn_layers', 1, 'number of layers in the top rnn module')
cmd:option('-if_bidirection', 0, 'if use bidirection on top rnn module')
-- for attention module
cmd:option('-att_sig_w', 2, 'the sigmoid function weight')
cmd:option('-att_sig_h', 0.25, 'the weight for the hidden summary')
cmd:option('-att_layers', 1, 'number of layers in the rnn of attention module')
cmd:option('-att_model', 'rnn', 'lstm or rnn')
cmd:option('-att_size', 64, 'size of rnn internal state (for attention model)')
--general setting
cmd:option('-TAGM_weight', 1, 'the weight for the TAGM weight')
cmd:option('-if_category', 0, 'if incorporate the item category transformation')
cmd:option('-category_way', 'mul', 'options: sum or mul')
cmd:option('-category_weight', 1, 'only for category_way==sum')
cmd:option('-if_bias', 0, 'if use bias term for each user and item')
cmd:option('-if_weighted_dot_product', 0, 'if use the weighted dot product in the top net')
cmd:option('-if_PReLU', 1, 'if use PReLU for the hidden representation')
cmd:option('-if_weighted_loss', 0)

--- optimization
cmd:option('-opt_method', 'rmsprop', 'the optimization method with options: 1. "rmsprop"  2. "gd" (exact gradient descent)')
cmd:option('-learning_rate',2e-3,'learning rate')
cmd:option('-learning_rate_decay',0.4,'learning rate decay')  
cmd:option('-learning_rate_decay_after',0,'in number of epochs, when to start decaying the learning rate')
cmd:option('-decay_rate',0.95,'decay rate for rmsprop')
cmd:option('-dropout',0.10,'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
cmd:option('-lambda', 0, 'the coefficient value for the regularization term')
cmd:option('-batch_size',64,'number of sequences to train on for gradient descent each time')
cmd:option('-max_epochs',200,'number of full passes through the training data')
cmd:option('-max_iterations',100000,'max iterations to run')                                              
cmd:option('-grad_clip',10,'clip gradients at this value')
cmd:option('-blowup_threshold',1e4,'the blowup threshold')
cmd:option('-check_gradient', false, 'whether to check the gradient value') 
cmd:option('-do_random_init', true, 'whether to initialize the parameters manually') 
-- for now we just perform on the training set, the more standard way should be on the validation set
cmd:option('-stop_iteration_threshold', 30,
       'if better than the later @ iterations , then stop the optimization')
cmd:option('-decay_threshold', 10, 'if better than the later @ iterations , then decay the learning rate')
cmd:option('-if_init_from_check_point', false, 'initialize network parameters from checkpoint at this path')
cmd:option('-if_direct_test_from_scratch', false)
cmd:option('-if_output_step_test_error', false)
cmd:option('-test_limit_number', 4000)
cmd:option('-learning_rate_threshold', 1e-5)
cmd:option('-if_ranking', 0)
cmd:option('-if_try_different_min_len', 1)

--- bookkeeping
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-print_every',1,'how many steps/minibatches between printing out the loss')
cmd:option('-evaluate_every',6400,'how many samples between evaluate the whole data set')
cmd:option('-checkpoint_dir', 'result', 'output directory where checkpoints get written')
cmd:option('-savefile','current_model','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
cmd:option('-accurate_gpu_timing',0,'set this flag to 1 to get precise timings when using GPU. Might make code bit slower but reports accurate timings.')
cmd:option('-disp_image', 1, 'if display image from the stn output')
cmd:option('-w1', 1, 'for disp_image window')
cmd:option('-w2', 1, 'for disp_image window')

--- GPU/CPU
-- currently, only supports CUDA
cmd:option('-gpuid',-1,'which gpu to use. -1 = use CPU')
cmd:text()

-- parse input params                                                                                                                                                                                                                                                                                                                                                                                                                                                   
local opt = cmd:parse(arg)

opt.user_module = opt.model_type 
opt.item_module = opt.model_type

-- decrease the learning rate proportionally
local factor = 32 / opt.rnn_size
opt.learning_rate = opt.learning_rate * factor

if opt.gpuid < 0 then                                                                                                   
  print('Perform calculation by CPU using the optimization method: ' .. opt.opt_method)  
  torch.setnumthreads(32)   
  print('number of thread: ', torch.getnumthreads())                                    
else
  print('Perform calculation by GPU with OpenCL using the optimization method: ' .. opt.opt_method)
end

torch.manualSeed(opt.seed)

-- about disp_image
if opt.disp_image then
  opt.w1 = nil
  opt.w2 = nil
end

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

if opt.if_direct_test_from_scratch then
  if opt.if_try_different_min_len == 0 then
    evaluate_process.evaluate_from_scratch(opt, false)
  elseif opt.if_try_different_min_len == 1 then
    evaluate_process.evaluate_from_scratch2(opt, false)
  end
else
  -- begin to train the model
  print('Begin to train the model...')
  train_process.train(opt)
  print("Training Done!")
  torch.manualSeed(opt.seed)
  evaluate_process.evaluate_from_scratch(opt, false)
end
