
local model_utils = require 'util/model_utils'

local TAGM = {}
TAGM.__index = TAGM

function TAGM.new(loader, opt)
  local self = {}
  setmetatable(self, TAGM)
  self:model(loader, opt)
  return self
end

local function recurrent_attention_gated_unit(input_size, rnn_size, n, dropout)
  dropout = dropout or 0 

  -- there will be n+2 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  table.insert(inputs, nn.Identity()()) -- attention_weight == input_gate == 1-forget_gate
  for L = 1,n do
    -- since we don't have output gate, hence we prev_c = prev_h
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  local x, input_size_L
  local outputs = {}
  local in_gate = inputs[2]
  local forget_gate = nn.AddConstant(1.0)(nn.MulConstant(-1.0)(in_gate))
  for L = 1,n do
    -- c,h from previos timesteps
    local prev_h = inputs[L+2]
    -- the input to this layer
    if L == 1 then 
      --      x = OneHot(input_size)(inputs[1])
      x = inputs[1]
      input_size_L = input_size
    else 
      x = outputs[L-1] 
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      input_size_L = rnn_size
    end
    -- evaluate the input sums at once for efficiency
    local i2h = nn.Linear(input_size_L, rnn_size)(x):annotate{name='i2h_'..L}
    local h2h = nn.Linear(rnn_size, rnn_size)(prev_h):annotate{name='h2h_'..L}
    local in_transform = nn.ReLU()(nn.CAddTable()({i2h, h2h}))
    -- decode the gates

    -- perform the LSTM update
    local next_h           = nn.CAddTable()({
      nn.CMulTable()({forget_gate, prev_h}),
      nn.CMulTable()({in_gate, in_transform})
    })
    table.insert(outputs, next_h)
  end

  -- set up the decoder
  local top_h = outputs[#outputs]
  local hidden_v = nn.Dropout(dropout)(top_h)
  table.insert(outputs, hidden_v)
  return nn.gModule(inputs, outputs)

end

function TAGM:model(loader, opt)

  local pre_m = nn.Replicate(opt.rnn_size, 1)
  self.pre_m = pre_m
  local class_n = loader.class_size
  local top_ragu, top_biragu
  top_ragu = recurrent_attention_gated_unit(loader.feature_dim, opt.rnn_size, opt.rnn_layers, opt.dropout)
  if opt.if_bidirection == 1 then
    top_biragu = recurrent_attention_gated_unit(loader.feature_dim, opt.rnn_size, opt.rnn_layers, opt.dropout)
  end
  -- the initial state of the cell/hidden states
  local rnn_init_state = {}
  for L=1,opt.rnn_layers do
    local h_init = torch.zeros(1, opt.rnn_size)
    if opt.gpuid >=0 then h_init = h_init:cuda() end
    table.insert(rnn_init_state, h_init:clone())
  end
  self.rnn_init_state = rnn_init_state
  local rnn_params_flat, rnn_grad_params_flat = top_ragu:getParameters()
  self.rnn = top_ragu
  self.birnn = top_biragu

  if opt.if_bidirection == 0 then
    self.params_size = rnn_params_flat:nElement()
  else
    self.params_size = 2*rnn_params_flat:nElement()
  end
  print('number of parameters in the top lstm model: ' .. self.params_size)
end

function TAGM:clone_model(max_len, opt) 
  print('cloning rnn')
  local clones_rnn = model_utils.clone_many_times(self.rnn, max_len)
  print('cloning ' .. max_len ..  ' rnns for each time series finished! ')
  self.clones_rnn = clones_rnn
  if opt.if_bidirection == 1 then
    print('cloning bidirectional rnn')
    local clones_birnn = model_utils.clone_many_times(self.birnn, max_len)
    print('cloning ' .. max_len ..  ' birnns for each time series finished! ')
    self.clones_birnn = clones_birnn
  end
end

function TAGM:init_params(opt)

end

function TAGM:forward(x, attention_weight, opt, flag)
  local pre_m = self.pre_m
  local lstm = self.rnn
  local rnn_init_state = self.rnn_init_state
  local clones_rnn = self.clones_rnn
  local clones_birnn = self.clones_birnn

  -- forward of pre_m
  local pre_out = pre_m:forward(attention_weight)
  -- forward of lstm
  local x_length = x:size(2)
  local init_state_global = clone_list(rnn_init_state)
  local rnn_input = x
  local rnn_state = {[0] = init_state_global}
  local hidden_z_value = nil  -- the value of rnn1 hidden unit in the last time step 
  -- we don't set the opt.seq_length, instead, we use the current length of the time series
  for t=1,x_length do
    if flag == 'test' then
      clones_rnn[t]:evaluate() -- make sure we are in correct mode (this is cheap, sets flag)
    else
      clones_rnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
    end
    local lst = clones_rnn[t]:forward{rnn_input:narrow(2, t, 1):squeeze(), pre_out:narrow(2, t, 1):squeeze(), unpack(rnn_state[t-1])}
    rnn_state[t] = {}
    for i=1,#rnn_init_state do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output
    -- last element is the output of the current time step: the hidden value after dropout  
    if t== x_length then  
      hidden_z_value = lst[#lst]
    end
  end
  self.rnn_state = rnn_state
  
  -- forward of bilstm
  if opt.if_bidirection == 1 then
    local birnn_state = {[x_length+1] = init_state_global}
    local bihidden_z_value = nil  -- the value of rnn1 hidden unit in the last time step 
    -- we don't set the opt.seq_length, instead, we use the current length of the time series
    for t=x_length, 1, -1 do
      if flag == 'test' then
        clones_birnn[t]:evaluate() -- make sure we are in correct mode (this is cheap, sets flag)
      else
        clones_birnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
      end
      local lst = clones_birnn[t]:forward{rnn_input:narrow(2, t, 1):squeeze(), pre_out:narrow(2, t, 1):squeeze(), unpack(birnn_state[t+1])}
      birnn_state[t] = {}
      for i=1,#rnn_init_state do table.insert(birnn_state[t], lst[i]) end -- extract the state, without output
      -- last element is the output of the current time step: the hidden value after dropout  
      if t== 1 then  
        bihidden_z_value = lst[#lst]
      end
    end
    self.birnn_state = birnn_state
    -- concatenate the output of forward and backward LSTM
    hidden_z_value = torch.cat(hidden_z_value, bihidden_z_value, 1)
  end
  
  self.hidden_z_value = hidden_z_value
  self.pre_out = pre_out
  
  return hidden_z_value
  
end

function TAGM:backward(x, attention_weight, opt, gradout)
  local pre_m = self.pre_m
  local lstm = self.rnn
  local rnn_init_state = self.rnn_init_state
  local clones_rnn = self.clones_rnn
  local clones_birnn = self.clones_birnn
  local rnn_state = self.rnn_state
  local birnn_state = self.birnn_state
  local x_length = x:size(2)

  local hidden_z_value = self.hidden_z_value
  local drnn_pre = torch.DoubleTensor(opt.rnn_size, x_length):zero()
  local bidrnn_pre = torch.DoubleTensor(opt.rnn_size, x_length):zero()
  if opt.gpuid >= 0 then drnn_pre = drnn_pre:cuda() end
  if opt.gpuid >= 0 then bidrnn_pre = bidrnn_pre:cuda() end
  local rnn_input = x
  local pre_out = self.pre_out
  local grad_net1, grad_net2
  if opt.if_bidirection == 1 then
    grad_net1 = gradout:sub(1, opt.rnn_size)
    grad_net2 = gradout:sub(opt.rnn_size+1, -1)
  else
    grad_net1 = gradout
  end
  local dzeros = torch.zeros(opt.rnn_size)
  if opt.gpuid >= 0 then dzeros = dzeros:cuda() end
  -- backward for rnn and birnn
  local drnn_state = {[x_length] = clone_list(rnn_init_state, true)} -- true also zeros the clones
  local drnn_x = torch.DoubleTensor(x_length, x:size(1))
  if opt.gpuid >= 0 then drnn_x = drnn_x:cuda() end
  -- perform back propagation through time (BPTT)
  for t = x_length,1,-1 do
    local doutput_t
    if t == x_length then
      doutput_t = grad_net1
    else
      doutput_t = dzeros
    end
    table.insert(drnn_state[t], doutput_t)
    local dlst = clones_rnn[t]:backward({rnn_input:narrow(2, t, 1):squeeze(), pre_out:narrow(2, t, 1):squeeze(), 
      unpack(rnn_state[t-1])}, drnn_state[t])
    drnn_state[t-1] = {}
    for k,v in pairs(dlst) do
      if k == 1 then
        drnn_x[t] = v
      elseif k == 2 then 
        -- note we do k-1 because first item is dembeddings, and then follow the 
        -- derivatives of the state, starting at index 2. I know...
        drnn_pre:select(2, t):copy(v)
      elseif k>2 then
        drnn_state[t-1][k-2] = v
      end
    end
  end
  
  if opt.if_bidirection == 1 then
    local bidrnn_state = {[1] = clone_list(rnn_init_state, true)} -- true also zeros the clones
    -- perform back propagation through time (BPTT)
    for t = 1, x_length do
      local doutput_t
      if t == 1 then
        doutput_t = grad_net2
      else
        doutput_t = dzeros
      end

      table.insert(bidrnn_state[t], doutput_t)
      local dlst = clones_birnn[t]:backward({rnn_input:narrow(2, t, 1):squeeze(), pre_out:narrow(2, t, 1):squeeze(), 
        unpack(birnn_state[t+1])}, bidrnn_state[t])
      bidrnn_state[t+1] = {}
      for k,v in pairs(dlst) do
        if k == 1 then
          drnn_x[t]:add(v)
        elseif k == 2 then 
          -- note we do k-1 because first item is dembeddings, and then follow the 
          -- derivatives of the state, starting at index 2. I know...
          bidrnn_pre:select(2, t):copy(v)
        elseif k>2 then
          bidrnn_state[t+1][k-2] = v
        end
      end
    end
    drnn_pre:add(bidrnn_pre)
  end
  
  -- backward for mul_net
  local grad_mul_net = pre_m:backward(attention_weight, drnn_pre)
  
  return grad_mul_net, drnn_x

end

return TAGM
