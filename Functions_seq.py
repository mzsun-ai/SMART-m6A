import torch
import os
from sklearn import metrics
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import Metrics
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.model_selection import KFold, StratifiedKFold


def get_distance_mask(L):

    m=np.zeros((3,L,L))


    for i in range(L):
        for j in range(L):
            for k in range(3):
                if abs(i-j)>0:
                    m[k,i,j]=1/abs(i-j)**(k+1)
    return m

def aug_data(df,aug_df):
    target_df = df.copy()
    new_df = aug_df[aug_df['id'].isin(target_df['id'])]
    ids= new_df.id.to_list()
    indices=[]
    for id in df.id:
        indices.append(ids.index(id))
    indices=np.asarray(indices)
    del target_df['structure']
    del target_df['predicted_loop_type']
    new_df = new_df.merge(target_df, on=['id','sequence'], how='left')

    df['cnt'] = df['id'].map(new_df[['id','cnt']].set_index('id').to_dict()['cnt'])
    df['log_gamma'] = 100
    df['score'] = 1.0
    df = new_df[df.columns]
    return df, indices


def MCRMSE(y_pred,y_true):
    colwise_mse = torch.mean(torch.square(y_true - y_pred), axis=1)
    MCRMSE = torch.mean(torch.sqrt(colwise_mse), axis=1)
    return MCRMSE

def weighted_MCRMSE(y_pred,y_true,ew):
    colwise_mse = torch.mean(ew*torch.square(y_true - y_pred), axis=1)
    MCRMSE = torch.mean(torch.sqrt(colwise_mse), axis=1)
    return MCRMSE

def get_train_val_indices(df, fold, SEED=2020, nfolds=5):
    splits = StratifiedKFold(n_splits=nfolds, random_state=SEED, shuffle=True)
    train_indices, val_indices = list(splits.split(df.sequence, df.label))[fold]
    return train_indices, val_indices

def get_train_val_indices_PL(data, fold, SEED=2020,nfolds=5):
    splits = KFold(n_splits=nfolds, random_state=SEED, shuffle=True)
    splits = list(splits.split(data))
    train_indices=splits[fold][0]
    val_indices=splits[fold][1]
    return train_indices,val_indices

def get_best_weights_from_fold(fold,top=5):
    csv_file = os.path.join('logs', f'pretrain_fold_{fold}.csv')

    history=pd.read_csv(csv_file)
    scores=np.asarray(-history.val_loss)
    top_epochs=scores.argsort()[-top:][::-1]
    print(scores[top_epochs])
    os.system('mkdir best_weights')

    for i in range(top):
        weights_path='checkpoints_fold{}/epoch{}.ckpt'.format(fold,history.epoch[top_epochs[i]])
        print(weights_path)
        os.system('cp {} best_weights/fold{}top{}.ckpt'.format(weights_path,fold,i+1))



def mutate_dna_sequence(sequence,nmute=15):
    perm = torch.randperm(sequence.size(1))
    to_mutate = perm[:nmute]
    mutation=torch.randint(4,size=(sequence.size(0),nmute))
    sequence[:,to_mutate]=mutation
    return sequence

def mutate_rna_input(sequence,nmute=.15):
    nmute=int(sequence.shape[1]*nmute)
    perm = torch.randperm(sequence.size(1))
    to_mutate = perm[:nmute]
    sequence_mutation=torch.randint(4,size=(sequence.size(0),nmute))
    structure_mutation=torch.randint(4,7,size=(sequence.size(0),nmute))
    d_mutation=torch.randint(7,14,size=(sequence.size(0),nmute))
    mutated=sequence.clone()
    mutated[:,to_mutate,0]=sequence_mutation
    mutated[:,to_mutate,1]=structure_mutation
    mutated[:,to_mutate,2]=d_mutation
    return mutated

def mask_rna_input(sequence,nmute=.15):
    nmute=int(sequence.shape[1]*nmute)
    perm = torch.randperm(sequence.size(1))
    to_mutate = perm[:nmute]
    masked=sequence.clone()
    masked[:,to_mutate,:]=14
    return masked

def get_MLM_mask(sequence,nmask=12):
    mask=np.zeros(sequence.shape,dtype='bool')
    to_mask=np.random.choice(len(sequence[0]),size=(nmask),replace=False)
    mask[:,to_mask]=True
    return mask


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        lr=param_group['lr']
    return lr

def save_weights(model,optimizer,epoch,folder):
    if os.path.isdir(folder)==False:
        os.makedirs(folder,exist_ok=True)
    torch.save(model.state_dict(), folder+'/epoch{}.ckpt'.format(epoch+1))



def validate(model,device,dataset,batch_size=64):
    #outputs=[]
    total_loss = 0
    total_samples = 0
    correct=0
    criterion = nn.BCEWithLogitsLoss()
    model.eval()
    with torch.no_grad():
        for data in tqdm(dataset):
            src = data['data'].to(device)
            bpp = data['bpp'].to(device)
            labels = data['labels'].float().to(device) 
            output = model(src, bpp, src_mask=data['src_mask'].to(device)).squeeze()
            #ew=data['ew'].to(device).float()
            #directions=data['directions']
            #directions=directions.reshape(len(directions),1)*np.ones(X.shape)
            #directions=torch.Tensor(directions).to(device).long()
            
                #outputs.append(model(X[:,i],bpps[:,5+i]+distance_mask)[:,:68])
            # print(outputs.shape)
            # exit()
            loss = criterion(output, labels) 
            #output=softmax(output)
            #classification_predictions = torch.argmax(output,dim=1).squeeze()
            #val_acc+=Metrics.accuracy(classification_predictions.cpu().numpy(),Y.cpu().numpy())
            #     predictions.append(pred.cpu().numpy())
            # for vector in output:
            #     outputs.append(vector.cpu().numpy())
            # for label in Y.cpu().numpy():
            #     ground_truths.append(label)
            #del output
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
            pred_labels = (torch.sigmoid(output) > 0.5).long()
            correct += (pred_labels == labels.long()).sum().item()
    #ground_truths=np.asarray(ground_truths,dtype='int')
    avg_loss = total_loss / total_samples
    acc = correct / total_samples
    print(f"Validation Loss: {avg_loss:.4f} | Accuracy: {acc:.4f}")
    return avg_loss


def revalidate(model,device,dataset,batch_size=64):
    batches=len(dataset)
    total=0
    predictions=[]
    #outputs=[]
    ground_truths=[]
    loss=0
    val_acc=0
    criterion=MCRMSE
    label_indices=[0,1,3]
    with torch.no_grad():
        for data in tqdm(dataset):
            X=data['data'].to(device,).long()[:,:]
            Y=data['labels'].to(device).float()
            bpps=data['bpp'].to(device).float()
            #directions=data['directions']
            #directions=directions.reshape(len(directions),1)*np.ones(X.shape)
            #directions=torch.Tensor(directions).to(device).long()
            output= model(X,bpps)[:,:68,]
            del X
            loss+=criterion(output,Y).mean()
            #output=softmax(output)
            classification_predictions = torch.argmax(output,dim=1).squeeze()
            val_acc+=Metrics.accuracy(classification_predictions.cpu().numpy(),Y.cpu().numpy())
            #     predictions.append(pred.cpu().numpy())
            # for vector in output:
            #     outputs.append(vector.cpu().numpy())
            # for label in Y.cpu().numpy():
            #     ground_truths.append(label)
            del output
    #ground_truths=np.asarray(ground_truths,dtype='int')
    torch.cuda.empty_cache()
    val_loss=(loss/batches).cpu()
    print(val_loss.item())
    return val_loss
