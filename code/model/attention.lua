
----
--given a time series, the attention model output a weight or a weighted vector for each time step of time series 

require 'nn'
local RNN = require 'model.my_RNN'
local LSTM = require 'model.my_LSTM'
local model_utils = require 'util.model_utils'
--require 'model.myMapTable'
require 'util/misc'

local Attention = {}
Attention.__index = Attention

function Attention.new(loader, opt)
  local self = {}
  setmetatable(self, Attention)
  self:model(loader, opt)
  return self
end

function Attention:model(loader, opt)

  local rnn_model = nil
  local birnn_model = nil --bidirectional rnn 
  if opt.att_model == 'lstm' then
    rnn_model = LSTM.lstm(loader.feature_dim, opt.att_size, opt.att_layers, opt.dropout)
  elseif opt.att_model == 'rnn' then
    rnn_model = RNN.rnn(loader.feature_dim, opt.att_size, opt.att_layers, opt.dropout)
  else
    error('no such attention model!')
  end
  if opt.att_model == 'lstm' then
    birnn_model = LSTM.lstm(loader.feature_dim, opt.att_size, opt.att_layers, opt.dropout)
  elseif opt.att_model == 'rnn' then
    birnn_model = RNN.rnn(loader.feature_dim, opt.att_size, opt.att_layers, opt.dropout)
  else
    error('no such attention model!')
  end

  -- the initial state of the cell/hidden states
  local rnn_init_state = {}
  for L=1,opt.att_layers do
    local h_init = torch.zeros(1, opt.att_size)
    if opt.gpuid >=0 then h_init = h_init:cuda() end
    table.insert(rnn_init_state, h_init:clone())
    if opt.att_model == 'lstm' then
      table.insert(rnn_init_state, h_init:clone())
    end
  end
  self.rnn_init_state = rnn_init_state
  self.rnn = rnn_model
  self.birnn = birnn_model
  -- ship the model to the GPU if desired
  if opt.gpuid >= 0 then rnn_model:cuda(); birnn_model:cuda()  end
  local rnn_params_flat, rnn_grad_params_flat = rnn_model:getParameters()
  self.params_size = rnn_params_flat:nElement()*2
  print('number of parameters in the attention model: ' .. self.params_size)

end

function Attention:clone_model(max_len)
  -- make a bunch of clones for input time series after flattening, sharing the same memory
  -- note that: it is only performed once for the reason of efficiency, 
  -- hence we clone the max length of times series in the data set for each rnn time series 
  print('cloning rnn')
  local clones_rnn = model_utils.clone_many_times(self.rnn, max_len)
  print('cloning ' .. max_len ..  ' rnns for each time series finished! ')
  local clones_birnn = nil
  print('cloning bidirectional rnn')
  clones_birnn = model_utils.clone_many_times(self.birnn, max_len)
  print('cloning ' .. max_len ..  ' birnns for each time series finished! ')
  self.clones_rnn = clones_rnn
  self.clones_birnn = clones_birnn
end

function Attention:init_lstm(opt)
  -- initialize the LSTM forget gates with slightly higher biases to encourage remembering in the beginning
  if opt.att_model == 'lstm' then
    for layer_idx = 1, opt.att_layers do
      for _,node in ipairs(self.rnn.forwardnodes) do
        if node.data.annotations.name == "i2h_" .. layer_idx then
          print('setting forget gate biases to 1 in LSTM layer ' .. layer_idx)
          -- the gates are, in order, i,f,o,g, so f is the 2nd block of weights
          node.data.module.bias[{{opt.att_size+1, 2*opt.att_size}}]:fill(1.0)
        end
      end
    end
  end 
  for layer_idx = 1, opt.att_layers do
    for _,node in ipairs(self.birnn.forwardnodes) do
      if node.data.annotations.name == "i2h_" .. layer_idx then
        print('setting forget gate biases to 1 in LSTM layer ' .. layer_idx)
        -- the gates are, in order, i,f,o,g, so f is the 2nd block of weights
        node.data.module.bias[{{opt.att_size+1, 2*opt.att_size}}]:fill(1.0)
      end
    end
  end
end

function Attention:forward(x, opt, flag)
  local rnn_init_state = self.rnn_init_state
  local x_length = x:size(2)
  local clones_rnn = self.clones_rnn
  local clones_birnn = self.clones_birnn

  local init_state_global = clone_list(rnn_init_state)
 
  -- perform forward for forward rnn
  local rnn_input = x
  local rnn_state = {[0] = init_state_global}
  local hidden_z_value = {}  -- the value of rnn1 hidden unit in the last time step 
  local hidden_summary = nil
  -- we don't set the opt.seq_length, instead, we use the current length of the time series
  for t=1,x_length do
    if flag == 'test' then
      clones_rnn[t]:evaluate() -- make sure we are in correct mode (this is cheap, sets flag)
    else
      clones_rnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
    end
    local lst = clones_rnn[t]:forward{rnn_input:narrow(2, t, 1):squeeze(), unpack(rnn_state[t-1])}
    rnn_state[t] = {}
    for i=1,#rnn_init_state do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output
    -- last element is the output of the current time step: the hidden value after dropout    
    hidden_z_value[t] = lst[#lst]
  end
  self.rnn_state = rnn_state
  hidden_summary = hidden_z_value[x_length]
  
  -- perform the forward pass for birnn: in the other direction
  local birnn_state, bihidden_z_value
  birnn_state = {[x_length+1] = init_state_global}
  bihidden_z_value = nil  -- the value of rnn1 hidden unit in the last time step 
  -- we don't set the opt.seq_length, instead, we use the current length of the time series
  for t=x_length, 1, -1 do
    if flag == 'test' then
      clones_birnn[t]:evaluate() -- make sure we are in correct mode (this is cheap, sets flag)
    else
      clones_birnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
    end
    local lst = clones_birnn[t]:forward{rnn_input:narrow(2, t, 1):squeeze(), unpack(birnn_state[t+1])}
    birnn_state[t] = {}
    for i=1,#rnn_init_state do table.insert(birnn_state[t], lst[i]) end -- extract the state, without output
    -- last element is the output of the current time step: the hidden value after dropout
    hidden_z_value[t] = torch.cat(hidden_z_value[t], lst[#lst], 1)
    if t == 1 then
      hidden_summary  = torch.cat(hidden_summary, lst[#lst], 1)
    end  
  end
  
  self.birnn_state = birnn_state
  return hidden_z_value, hidden_summary
end

function Attention:backward(opt, grad_h, grad_summary, x, if_interact)
  local rnn_init_state = self.rnn_init_state
  local clones_rnn = self.clones_rnn
  local clones_birnn = self.clones_birnn
  local rnn_state = self.rnn_state
  local birnn_state = self.birnn_state
  
  local x_length = x:size(2)
  local rnn_input = x
  local drnn_x = torch.DoubleTensor(x_length, x:size(1))
  if opt.gpuid >= 0 then drnn_x = drnn_x:cuda() end
  -- backward for rnn and birnn
  local drnn_state = {[x_length] = clone_list(rnn_init_state, true)} -- true also zeros the clones
  -- perform back propagation through time (BPTT)
  for t = x_length,1,-1 do
    local doutput_t
    if if_interact and t == x_length then
      doutput_t = torch.add(grad_h[t]:sub(1, opt.att_size), grad_summary:sub(1, opt.att_size))
    else
      doutput_t = grad_h[t]:sub(1, opt.att_size)
    end
    table.insert(drnn_state[t], doutput_t)
    local dlst = clones_rnn[t]:backward({rnn_input:narrow(2, t, 1):squeeze(), unpack(rnn_state[t-1])}, drnn_state[t])
    drnn_state[t-1] = {}
    for k,v in pairs(dlst) do
      if k== 1 then
        drnn_x[t] = v
      elseif k ~= 1 then -- k == 1 is gradient on x, which we dont need
        drnn_state[t-1][k-1] = v
      end
    end
  end
  -- backward for birnn
  local bidrnn_state = {[1] = clone_list(rnn_init_state, true)} -- true also zeros the clones
  -- perform back propagation through time (BPTT)
  for t = 1, x_length do
    local doutput_t
    if if_interact and t==1 then
      doutput_t= torch.add(grad_h[t]:sub(opt.att_size+1, -1), grad_summary:sub(opt.att_size+1, -1))
    else
      doutput_t= grad_h[t]:sub(opt.att_size+1, -1)
    end
    table.insert(bidrnn_state[t], doutput_t)
    local dlst = clones_birnn[t]:backward({rnn_input:narrow(2, t, 1):squeeze(), unpack(birnn_state[t+1])}, bidrnn_state[t])
    bidrnn_state[t+1] = {}
    for k,v in pairs(dlst) do
      if k == 1 then
        drnn_x[t]:add(v)
      elseif k ~= 1 then -- k == 1 is gradient on x, which we dont need
        bidrnn_state[t+1][k-1] = v
      end
    end
  end
  
  return drnn_x
end

return Attention
