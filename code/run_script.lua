
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

local single_run = require 'single_run'

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
cmd:option('-rnn_size', 64, 'size of LSTM internal state (for top lstm model)')
cmd:option('-rnn_layers', 1, 'number of layers in the top LSTM')
cmd:option('-if_bidirection', 0, 'if use bidirection on top rnn')
-- for attention module
cmd:option('-att_sig_w', 2, 'the sigmoid function weight')
cmd:option('-att_sig_h', 0.25, 'the weight for the hidden summary')
cmd:option('-att_layers', 1, 'number of layers in the LSTM')
cmd:option('-att_model', 'rnn', 'lstm or rnn')
cmd:option('-att_size', 64, 'size of LSTM internal state (for attention model)')
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
cmd:option('-dropout',0.0,'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
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
cmd:option('-if_train_validation', false)
cmd:option('-if_output_step_test_error', false)
cmd:option('-test_limit_number', 4000)
cmd:option('-learning_rate_threshold', 1e-5)
cmd:option('-if_ranking', 0)
cmd:option('-if_try_different_min_len', 0)

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
opt.original_learning_rate = opt.learning_rate

-----------------------------begin to run experments-----------------------------
----- TAGM-TAGM
--local model_modes = {}
----model_modes[#model_modes+1] = {'IARN', 'IARN'}
--model_modes[#model_modes+1] = {'TAGM', 'TAGM'}
----model_modes[#model_modes+1] = {'rnn', 'rnn'}
----model_modes[#model_modes+1] = {'lstm', 'lstm'}
----model_modes[#model_modes+1] = {'TAGM', 'rnn'}
----model_modes[#model_modes+1] = {'rnn', 'TAGM'} 

opt.user_module = opt.model_type 
opt.item_module = opt.model_type

local model_modes = {}
model_modes[#model_modes+1] = {opt.user_module, opt.item_module}

--- rnn_size
local rnn_sizes = 64

--- if_category
--local category_mode = {1, 0}
local category_mode = {0}

--- if_negative
--local negative_mode = {0, 1}
local negative_mode = {0}

--- if weight_dot_product
--local weight_dot_product_mode = {0, 1}

--- dropout
local dropouts = {0.0, 0.25, 0.5}
for mod = 1, #model_modes do
  opt.user_module = model_modes[mod][1]
  opt.item_module = model_modes[mod][2]
  if opt.user_module == 'IARN' and opt.item_module == 'IARN' then
    local if_drop = false
    local best_drop = nil
    opt.rnn_size = rnn_sizes
    opt.att_size = rnn_sizes
    for neg = 1, #negative_mode do
      opt.if_negative = negative_mode[neg]
      for cate = 1, #category_mode do
        opt.if_category = category_mode[cate]
        if not if_drop then 
          local errs = torch.Tensor(#dropouts)
          for drop = 1, #dropouts do
            opt.dropout = dropouts[drop]
            
            errs[drop] = single_run.run_single_experiment(opt)
          end
          local v, ind = torch.sort(errs)
          best_drop = dropouts[ind[1]]
          if_drop = true
          opt.dropout = best_drop
        else
          opt.dropout = best_drop
          local cerr = single_run.run_single_experiment(opt)
        end
      end
    end
  else
    local if_drop = false
    local best_drop = nil
    opt.rnn_size = rnn_sizes
    opt.att_size = rnn_sizes
    opt.if_negative = 0
    opt.if_category = 0
    if not if_drop then 
      local errs = torch.Tensor(#dropouts)
      for drop = 1, #dropouts do
        opt.dropout = dropouts[drop]
        errs[drop] = single_run.run_single_experiment(opt)
      end
      local v, ind = torch.sort(errs)
      best_drop = dropouts[ind[1]]
      if_drop = true
      opt.dropout = best_drop
    else
      opt.dropout = best_drop
      local cerr = single_run.run_single_experiment(opt)
    end
  end
end



