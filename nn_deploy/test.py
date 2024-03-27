import torch
from torch import fx

def func_to_trace(x):
    if x.sum() > 0:
        return torch.relu(x)
    else:
        return torch.neg(x)

def func_to_trace1(x):
    return torch.neg(x)


traced = torch.fx.symbolic_trace(func_to_trace1)
print(traced)
"""
  <...>
  File "dyn.py", line 6, in func_to_trace
    if x.sum() > 0:
  File "pytorch/torch/fx/proxy.py", line 155, in __bool__
    return self.tracer.to_bool(self)
  File "pytorch/torch/fx/proxy.py", line 85, in to_bool
    raise TraceError('symbolically traced variables cannot be used as inputs to control flow')
torch.fx.proxy.TraceError: symbolically traced variables cannot be used as inputs to control flow
"""

def f(x, flag):
    if flag: return x
    else: return x*2

# fx.symbolic_trace(f) # Fails!

fx_traced = fx.symbolic_trace(f, concrete_args={'flag': True})
fx_traced.print_readable()