import torch
import pickle
import torch.fx as fx
from .fx_utils import CustomTracer
import os

def model2gm(model: torch.nn.Module) -> fx.graph_module.GraphModule:
    trace = CustomTracer()
    fx_trace = trace.trace(model)
    fx_model = fx.graph_module.GraphModule(trace.root, fx_trace)
    return fx_model

def save(model,name,weight="none"):
    assert ".pickle" in name,"please save example.pickle"
    weight=name.replace("pickle","pth") if weight=="none" else weight
    model.eval()
    model=model2gm(model)
    with open(name, "wb") as file:
        pickle.dump(model.cpu(), file)
    torch.save(model.state_dict(), weight)
    print("save success")

def load(name,weight="none"):
    assert ".pickle" in name,"please save example.pickle"
    weight=name.replace("pickle","pth") if weight=="none" else weight
    with open(name, "rb") as file:
        model_test=pickle.load(file)
    if os.path.exists(weight):
        model_test.load_state_dict(torch.load(weight))
        print("load success")
    else:
        print("no weigth load")
    model_test.eval()
    model_test.to("cpu")
    return model_test
