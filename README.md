# IARN
This is the implementation for the model 'Interacting Attention-gated Recurrent Networks (IARN)', which is proposed in the paper:   
Wenjie Pei\*, Jie Yang\*, Zhu Sun, Jie Zhang, Alessandro Bozzon and David M.J. Tax (\*both authors contributed equally).
[Interacting Attention-gated Recurrent Networks for Recommendation](https://dl.acm.org/citation.cfm?id=3133005).
ACM International Conference on Information and Knowledge Management (__CIKM__), __full__ paper, 2017.

The code is implemented in Lua and Torch. It contains mainly the following parts:  
* main.lua:   the starting point of the entire code. 
* run_script.lua: another starting point of the entire code, which is script to run a batch of experiments together.
* train_process.lua: the training process.
* evaluate_process.lua: the evaluation process. 
* package 'model' contains the required models including attention model, TAGM, LSTM, GRU and plain-RNN. 
* package 'util' contains the required small utilities such as data loader. 
