-- internal library
local optim = require 'optim'
local path = require 'pl.path'

-- local library
local RNN = require 'model.my_RNN'
local model_utils = require 'util.model_utils'
local data_loader = require 'util.data_loader'
require 'util.misc'
local Top_Net = require 'model/Top_NN_Classifier'
local evaluate_process = require 'evaluate_process'
local table_operation = require 'util/table_operation'
local define_my_model = require 'model/define_my_model'

local train_process = {}

--- process one batch to get the gradients for optimization and update the parameters 
-- return the loss value of one minibatch of samples
local function feval(opt, loader, model, rmsprop_para, iter_count)
  -- decode the model and parameters, 
  -- since it is just the reference to the same memory location, hence it is not time-consuming.

  local category_transform = model.category_transform
  local user_attention = model.user_attention
  local user_att_weight_net = model.user_att_weight
  local item_attention = model.item_attention
  local item_att_weight_net = model.item_att_weight
  local rnn_user = model.rnn_user
  local rnn_item = model.rnn_item
  local top_net = model.top_net
  local MF_net = model.MF_net
  local criterion = model.criterion 
  local params_flat = model.params_flat
  local grad_params_flat = model.grad_params_flat
  local params_grad_all_batches = model.params_grad_all_batches
  local bias_term = model.Bias_Term

  ---------------------------- get minibatch --------------------------
  ---------------------------------------------------------------------

  local data_index = loader:get_next_train_batch(opt.batch_size)
  local loss_total = 0
  params_grad_all_batches:zero()

  -- Process the batch of samples one by one, since different sample contains different length of time series, 
  -- hence it's not convenient to handle them together
  for batch = 1, opt.batch_size do
    local current_data_index = data_index[batch]
    local train_pair = loader.train_pairs[current_data_index]
    local user_x = loader.user_X[train_pair[1]+1]
    local item_x = loader.item_X[train_pair[2]+1]
    local true_y = loader.train_T[current_data_index]
    local user_length = user_x:size(2)
    local item_length = item_x:size(2)
    if opt.gpuid >= 0 then 
      user_x = user_x:cuda()
      item_x = item_x:cuda()
      true_y = true_y:cuda() 
    end
    
    
    ---------------------- forward pass of the whole model -------------------  
    --------------------------------------------------------------------------  

    -- perform the forward pass for the category transform for item time series
    local cate_item_x = nil
    local category = nil
    if opt.if_category == 1 then
      category = loader.item_category[train_pair[2]+1]
    end
--    print('category:')
--    print(category)
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
      user_hidden_z_value, user_hidden_summary = user_attention:forward(user_x, opt, 'training')
      item_hidden_z_value, item_hidden_summary = item_attention:forward(cate_item_x, opt, 'training')
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
  
    -- perform the forward for the rnn_user and rnn_item
    local user_hidden = rnn_user:forward(user_x, user_att_weights, opt, 'training')
    local item_hidden = rnn_item:forward(cate_item_x, item_att_weights, opt, 'training')
    -- perform the forward for the top-net module
    local rnn_net_output = top_net:forward({user_hidden, item_hidden})
    -- perform the forward for the MF
    local one = torch.ones(1)
    if opt.gpuid >= 0 then one = one:cuda() end
    local MF_output = MF_net:forward({one, one:clone()})
    -- perform the forward for the bias term
    local bias_out = nil
    if opt.if_bias == 1 then
      bias_out = bias_term:forward(train_pair[1]+1, train_pair[2]+1)
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
    loss_total = loss_total + current_loss


    ---------------------- backward pass of the whole model ---------------------
    -----------------------------------------------------------------------------

    -- peform the backprop on the top_net
    grad_params_flat:zero()
    local grad_criterion = nil
    if opt.if_bias == 1 then
      grad_criterion = criterion:backward({rnn_net_output+bias_out, MF_output}, {true_y, true_y})
      if opt.if_weighted_loss == 1 then
        grad_criterion[1] = grad_criterion[1] * loader.class_weight[true_y:squeeze()]
        grad_criterion[2] = grad_criterion[2] * loader.class_weight[true_y:squeeze()]
      end
      bias_term:backward(train_pair[1]+1, train_pair[2]+1, grad_criterion[1])
    else
      grad_criterion = criterion:backward({rnn_net_output, MF_output}, {true_y, true_y})
      if opt.if_weighted_loss == 1 then
        grad_criterion[1] = grad_criterion[1] * loader.class_weight[true_y:squeeze()]
        grad_criterion[2] = grad_criterion[2] * loader.class_weight[true_y:squeeze()]
      end
    end
    local grad_rnn_top_net = top_net:backward({user_hidden, item_hidden}, grad_criterion[1])
    local grad_MF_output = MF_net:backward({one, one:clone()}, grad_criterion[2])    
    local grad_rnn_user = rnn_user:backward(user_x, user_att_weights, opt, grad_rnn_top_net[1])
    local grad_rnn_item, item_dx = rnn_item:backward(cate_item_x, item_att_weights, opt, grad_rnn_top_net[2])
    if opt.user_module == 'TAGM' or opt.user_module == 'IARN' then
      local if_interact = false
      if opt.user_module == 'IARN' then if_interact = true end
      local user_grad_h, item_grad_s = user_att_weight_net:backward(user_hidden_z_value, item_hidden_summary, grad_rnn_user, opt)
      local item_grad_h, user_grad_s = item_att_weight_net:backward(item_hidden_z_value, user_hidden_summary, grad_rnn_item, opt)
      user_attention:backward(opt, user_grad_h, user_grad_s, user_x, if_interact)
      local item_d = item_attention:backward(opt, item_grad_h, item_grad_s, cate_item_x, if_interact)
      item_dx:add(item_d)
    end
    
    if opt.if_category == 1 and category:nElement() > 0 then
      category_transform.backward(category, item_x:t(), item_dx)
    end
    params_grad_all_batches:add(grad_params_flat)

    -- for gradient check
    if opt.check_gradient then
      evaluate_process.grad_check(model, user_x, item_x, true_y, opt, category, train_pair, loader)
      print('\n')
      if batch > 1 then os.exit() end
    end
--    collectgarbage()
  end
  loss_total = loss_total / opt.batch_size
  -- udpate all the parameters
  params_grad_all_batches:div(opt.batch_size)
  params_grad_all_batches:clamp(-opt.grad_clip, opt.grad_clip)
  if opt.opt_method == 'rmsprop' then
    local function feval_rmsprop(p)
      return loss_total, params_grad_all_batches
    end
    optim.rmsprop(feval_rmsprop, params_flat, rmsprop_para.config)
  elseif opt.opt_method == 'gd' then -- 'gd' simple direct minibatch gradient descent
    params_flat:add(-opt.learning_rate, params_grad_all_batches)
  else
    error("there is no such optimization option!")  
  end

  return loss_total
end

--- major function 
function train_process.train(opt)

  torch.manualSeed(opt.seed)
  ------------------- create the data loader class ----------
  -----------------------------------------------------------

  local loader = data_loader.create(opt)
  local do_random_init = true

  ------------------ begin to define the whole model --------------------------
  -----------------------------------------------------------------------------
  local model = {}
  model, opt = define_my_model.define_model(opt, loader)


  --------------- start optimization here -------------------------
  -----------------------------------------------------------------
  -- for rmsprop
  local rmsprop_config = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
  local rmsprop_state = {}
  local rmsprop_para = {config = rmsprop_config, state = rmsprop_state}

  local iterations = math.floor(opt.max_epochs * loader.nTrain / opt.batch_size)
  local iterations_per_epoch = math.floor(loader.nTrain / opt.batch_size)
  local train_losses = torch.zeros(iterations)
  local timer = torch.Timer()
  local time_s = timer:time().real
  local epoch = 0
  local better_times_total = 0
  local better_times_decay = 0
  local current_best_err = 1e10
  for i = 1, iterations do
    if epoch > opt.max_epochs then break end
    if i>opt.max_iterations then break end
    epoch = i / loader.nTrain * opt.batch_size
    local time_ss = timer:time().real
    -- optimize one batch of training samples
    train_losses[i] = feval(opt, loader, model, rmsprop_para, i)
    local time_ee = timer:time().real
    local time_current_iteration = time_ee - time_ss
    if i % 1 == 0 then collectgarbage() end

    -- handle early stopping if things are going really bad
    local function isnan(x) return x ~= x end
    if isnan(train_losses[i]) then
      print('loss is NaN.  This usually indicates a bug.' .. 
        'Please check the issues page for existing issues, or create a new issue, ' .. 
        'if none exist.  Ideally, please state: your operating system, 32-bit/64-bit, your blas version, cpu/cuda/cl?')
      break -- halt
    end
    -- check if the loss value blows up
    local function is_blowup(loss_v)
      if loss_v > opt.blowup_threshold then
        print('loss is exploding, aborting:', loss_v)
        return true
      else 
        return false
      end
    end
    if is_blowup(train_losses[i]) then
      break
    end

    if i % opt.print_every == 0 then
      print(string.format("%d/%d (epoch %.3f), train_loss = %6.8f, time/batch = %.4fs", 
        i, iterations, epoch, train_losses[i], time_current_iteration))
    end

    if i * opt.batch_size % opt.evaluate_every == 0 then
      local temp_sum_loss = torch.sum(train_losses:sub(i - opt.evaluate_every/opt.batch_size+1, i))
      local temp_mean_loss = temp_sum_loss / opt.evaluate_every * opt.batch_size
      print(string.format('average loss in the last %d iterations = %6.8f', opt.evaluate_every, temp_mean_loss))
      print('learning rate: ', opt.learning_rate)

      local whole_validation_loss,  validation_err = nil
      if opt.validation_size == 0 then
        local whole_train_loss, train_err = evaluate_process.evaluate_set('train', opt, loader, model)
        whole_validation_loss = whole_train_loss
        validation_err = train_err
      else
        whole_validation_loss,  validation_err = evaluate_process.evaluate_set('validation', opt, loader, model)
      end
      local whole_test_loss, test_acc 
      if opt.if_output_step_test_error then
        whole_test_loss, test_acc = evaluate_process.evaluate_set('test', opt, loader, model, true)
      end
      local time_e = timer:time().real
      print(string.format('elasped time in the last %d iterations: %.4fs,    total elasped time: %.4fs', 
        opt.evaluate_every, time_e-time_s, time_e))
      if validation_err < current_best_err then
        current_best_err = validation_err
        better_times_total = 0
        better_times_decay = 0
        --- save the current trained best model
        define_my_model.save_model(opt, model)
        if validation_err == 0 then
          break
        end
      else
        better_times_total = better_times_total + 1
        better_times_decay = better_times_decay + 1
        if better_times_total >= opt.stop_iteration_threshold or opt.learning_rate < opt.learning_rate_threshold then
          print(string.format('no more better result in %d iterations! hence stop the optimization!', 
            opt.stop_iteration_threshold))
          break
        elseif better_times_decay >= opt.decay_threshold then
          print(string.format('no more better result in %d iterations! hence decay the learning rate', 
            opt.decay_threshold))
          local decay_factor = opt.learning_rate_decay
          rmsprop_config.learningRate = rmsprop_config.learningRate * decay_factor -- decay it
          opt.learning_rate = rmsprop_config.learningRate -- update the learning rate in opt
          print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. rmsprop_config.learningRate)
          if opt.learning_rate < opt.learning_rate_threshold then
            print(string.format('no more better result in %d iterations! hence stop the optimization!', 
              opt.stop_iteration_threshold))
            break
          end
          better_times_decay = 0 
          -- back to the currently optimized point
          print('back to the currently best optimized point...')
          model = define_my_model.load_model(opt, model, false)
        end
      end     
      print('better times: ', better_times_total, '\n\n')
      -- save to log file
      local temp_file = nil
      if i  == 1 and not opt.if_init_from_check_point then
        temp_file = io.open(string.format('%s/%s_results_GPU_%d_dropout_%1.2f_negative_%d.txt',
          opt.current_result_dir, opt.opt_method, opt.gpuid, opt.dropout, opt.if_negative), "w")
      else
        temp_file = io.open(string.format('%s/%s_results_GPU_%d_dropout_%1.2f_negative_%d.txt', 
          opt.current_result_dir, opt.opt_method, opt.gpuid, opt.dropout, opt.if_negative), "a")
      end
      temp_file:write('better times: ', better_times_total, '\n')
      temp_file:write('learning rate: ', opt.learning_rate, '\n')
      temp_file:write(string.format("%d/%d (epoch %.3f) \n", i, iterations, epoch))
      temp_file:write(string.format('average loss in the last %d (%5d -- %5d) iterations = %6.8f \n', 
        opt.evaluate_every/opt.batch_size, i-opt.evaluate_every/opt.batch_size+1, i, temp_mean_loss))
      --      temp_file:write(string.format('train set loss = %6.8f, train age mean absolute error= %6.8f\n', 
      --       whole_train_loss, differ_avg_train ))
      temp_file:write(string.format('validation set loss = %6.8f, validation accuracy= %6.8f\n', 
        whole_validation_loss, validation_err ))
      if opt.if_output_step_test_error then
        temp_file:write(string.format('test set loss = %6.8f, test accuracy = %6.8f\n', 
          whole_test_loss, test_acc ))
      end
      temp_file:write(string.format('elasped time in the last %d iterations: %.4fs,    total elasped time: %.4fs\n', 
        opt.evaluate_every, time_e-time_s, time_e))
      temp_file:write(string.format('\n'))
      temp_file:close()
      time_s = time_e
    end
  end
  local time_e = timer:time().real
  print('total elapsed time:', time_e)
end

return train_process
    
    