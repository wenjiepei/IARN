
local my_criterion = {}


function my_criterion.criterion(TAGM_weight)
  local criterion_TAGM = nn.MSECriterion()
  local criterion_fixed = nn.MSECriterion()
  my_criterion.TAGM_weight = TAGM_weight
  local criterion_all = nn.ParallelCriterion():add(criterion_TAGM, my_criterion.TAGM_weight):add(criterion_fixed, 1-my_criterion.TAGM_weight)
  my_criterion.criterion_all = criterion_all
  return criterion_all
end

function my_criterion:set_weight(TAGM_weight)
  assert(#(self.criterion_all.weights)==2)
  self.criterion_all.weights[1] = TAGM_weight
  self.criterion_all.weights[2] = 1-TAGM_weight 
end

function my_criterion:get_TAGM_weight()
  assert(#(self.criterion_all.weights)==2)
  return self.criterion_all.weights[1]
end

return my_criterion