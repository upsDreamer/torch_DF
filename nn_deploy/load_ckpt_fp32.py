import os
import sys
import torch

sys.path.append("./AndesAIRE-NN-SDK/NNPilot")

from common.fx_utils import andes_preprocessing, add_idd

def return_fp32_model():
    with torch.no_grad():
        # import pdb; pdb.set_trace()
        print("deepfilternet:")
        torch_model_path=sys.argv[1]
        model_test = torch.load(torch_model_path)
        model_test=andes_preprocessing(model_test)
        model_test=add_idd(model_test)
    model_test.eval()
    print(model_test)
    return model_test

return_fp32_model()
