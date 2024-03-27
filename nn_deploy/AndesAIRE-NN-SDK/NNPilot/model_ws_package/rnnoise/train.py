import torch
from tqdm import tqdm
from .evaluation import QuantStub
from .rnnoise_pre_processing import RNNoisePreProcess
import numpy as np
def chunks(in_list, n):
    new_list=[]
    res=[]
    for i in range(0, len(in_list), n):
        if len(in_list[i:i + n])<n:
            res=in_list[i:i + n]
        else:
            new_list+=in_list[i:i + n]
    return new_list,res

def my_mask(y_true):
    """Used to mask off gain values if label indicates no audible signal.

    Ref: https://github.com/xiph/rnnoise/blob/1cbdbcf1283499bbb2230a6b0f126eb9b236defd/training/rnn_train.py#L34
    """
    return torch.minimum(y_true+1,y_true-y_true+1)

def my_cost(y_pred, y_true, bce):
    """Custom loss function for training noise reduction output.

    Ref: https://github.com/xiph/rnnoise/blob/1cbdbcf1283499bbb2230a6b0f126eb9b236defd/training/rnn_train.py#L40."""
    return torch.mean(my_mask(y_true) *(10*torch.square(torch.square(torch.sqrt(y_pred) - torch.sqrt(y_true)))
                                     + torch.square(torch.sqrt(y_pred) - torch.sqrt(y_true))
                                     + 0.01*bce(y_pred, y_true)))

def my_crossentropy(y_pred,y_true,bce):
    """Cross entropy loss for vad output.
    0 if y_true label is 0.5 - meaning we aren't sure if speech is present.

    Ref:https://github.com/xiph/rnnoise/blob/1cbdbcf1283499bbb2230a6b0f126eb9b236defd/training/rnn_train.py#L31"""
    return torch.mean(2*torch.abs(y_true-0.5) * bce(y_pred, y_true))

def training_set(model,device):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.00001)#lr=0.00015
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, nesterov=True)
    scheduler={}
    criterion={}
    return criterion,optimizer,scheduler

def train_one_epoch(model,dataloaders,data_config,criterion,optimizer,scheduler,device,symm=True,bits=8):
    model.to(device)
    model.train()
    running_loss = 0.0
    running_corrects = 0
    running_size=0
    res_features=[]
    res_labels=[]
    res_vad=[]
    vad_gru_state = torch.zeros((1, 24)).to(device)
    noise_gru_state = torch.zeros((1, 48)).to(device)
    denoise_gru_state = torch.zeros((1, 96)).to(device)
    batch_size=32
    chunk_len=1
    bce=torch.nn.BCELoss()
    bce2=torch.nn.BCELoss()
    optimizer.zero_grad()
    total_iter=0
    update_time=0
    tmp=0
    for clean_speech_audio, noise_audio in tqdm(dataloaders):
        preprocess = RNNoisePreProcess(training=True)
        tmp=tmp+1
        num_samples = len(noise_audio) // preprocess.FRAME_SIZE
        input_features, output_labels, vad_labels = preprocess.get_training_features(clean_speech_audio,
                                                                                 noise_audio,
                                                                                 len(noise_audio)//1000)
        #"concate with previous data"
        this_file=len(input_features)
        input_features=res_features+input_features
        output_labels=res_labels+output_labels
        vad_labels=res_vad+vad_labels

        input_features,res_features=chunks(input_features,chunk_len)
        output_labels,res_labels=chunks(output_labels,chunk_len)
        vad_labels,res_vad=chunks(vad_labels,chunk_len)
        # a valid batch_size input

        if len(input_features)>0:
            input_data=np.array(input_features).reshape((-1,chunk_len,42))
            output_labels=np.array(output_labels).reshape((-1,chunk_len,22))
            vad_labels=np.array(vad_labels).reshape((-1,chunk_len,1))
            it,b,_=input_data.shape
            if update_time>=500000:
                return model
            for idx in range(it):
                total_iter=total_iter+1
                input_data_=torch.tensor(input_data[idx],dtype=torch.float).unsqueeze(1).to(device)
                output_labels_=torch.tensor(output_labels[idx],dtype=torch.float).to(device)
                vad_labels_=torch.tensor(vad_labels[idx],dtype=torch.float).to(device)
                QuantStub(input_data_,data_config['fp32_min'],data_config['fp32_max'],symm,bits,isHW=False)
                denoise_gru_state, denoise_output, noise_gru_state, vad_gru_state, vad_out = model(input_data_,vad_gru_state,noise_gru_state,denoise_gru_state)
                loss_out=my_cost(denoise_output,output_labels_.unsqueeze(1),bce)
                loss_vad=my_crossentropy(vad_out,vad_labels_.unsqueeze(1),bce2)
                loss=10.0*loss_out+0.5*loss_vad
                loss.backward()
                if torch.isnan(loss):
                    optimizer.zero_grad()
                    vad_gru_state=vad_gru_state.squeeze(1).detach()
                    denoise_gru_state = denoise_gru_state.squeeze(1).detach()
                    noise_gru_state = noise_gru_state.squeeze(1).detach()
                    vad_gru_state = torch.zeros((1, 24)).to(device)
                    noise_gru_state = torch.zeros((1, 48)).to(device)
                    denoise_gru_state = torch.zeros((1, 96)).to(device)
                    continue
                if total_iter % batch_size==0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    update_time += 32
                    vad_gru_state=vad_gru_state.squeeze(1).detach()
                    denoise_gru_state = denoise_gru_state.squeeze(1).detach()
                    noise_gru_state = noise_gru_state.squeeze(1).detach()
                    vad_gru_state = torch.zeros((1, 24)).to(device)
                    noise_gru_state = torch.zeros((1, 48)).to(device)
                    denoise_gru_state = torch.zeros((1, 96)).to(device)

                else:
                    vad_gru_state=vad_gru_state.squeeze(1).detach()
                    denoise_gru_state = denoise_gru_state.squeeze(1).detach()
                    noise_gru_state = noise_gru_state.squeeze(1).detach()
    return model
