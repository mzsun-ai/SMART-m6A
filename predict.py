import os, argparse, json, pickle, numpy as np, pandas as pd, torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from Dataset_seq   import RNADataset    
from Network_seq  import SMARTM6A   

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu',      default='1', help='CUDA_VISIBLE_DEVICES')
    p.add_argument('--test',     default='/data/uestc/sunmingze/SMARTM6A/test_deepsramp/test.jsonl')
    p.add_argument('--path',     default='/data/uestc/sunmingze/SMARTM6A/test_deepsramp/', help=' bpps/ dotbrackets/ loops/ ')
    p.add_argument('--weights',  nargs='+',   required=True,
                   help=' ckpt')
    p.add_argument('--batch',    type=int, default=32)
    p.add_argument('--out',      default='submission.csv')
    p.add_argument('--ntoken', type=int, default=15)
    p.add_argument('--nclass', type=int, default=1)
    p.add_argument('--ninp',   type=int, default=256)
    p.add_argument('--nhead',  type=int, default=32)
    p.add_argument('--nhid',   type=int, default=1024)
    p.add_argument('--nlayers',type=int, default=5)
    p.add_argument('--dropout',type=float,default=0.1)
    p.add_argument('--kmers',  type=int, nargs='+', default=[2,3,4,5,6])
    p.add_argument('--stride', type=int, default=1)
    return p.parse_args()

@torch.no_grad()
def predict_one_model(model, loader, device, attn_save_dir=None):
    model.eval()
    probs, ids, labels_all = [], [], []

    for batch in tqdm(loader, desc='inference'):
        src  = batch['data'].to(device)      # (B, L, 3)
        bpp  = batch['bpp'].to(device)       # (B, 4, L, L)
        logit, attn_dict = model(src, bpp)   # (B,)
        logit = logit.squeeze(-1)
        prob = torch.sigmoid(logit).cpu().numpy()
        probs.append(prob)

        ids  += batch['id']
        labels_all += batch['labels'].cpu().numpy().tolist()

    #   
    #     if attn_save_dir is not None:
    #         attn_per_layer = attn_dict["encoder"]
    #         batch_size = src.size(0)

    #         for layer_idx, attn_tensor in enumerate(attn_per_layer):
    #             if layer_idx != 4:  
    #                 continue

    #             # attn_tensor: (B, nhead, L, L)
    #             layer_dir = os.path.join(attn_save_dir, f"layer{layer_idx}")
    #             os.makedirs(layer_dir, exist_ok=True)

    #             for i in range(batch_size):
    #                 sample_id = batch['id'][i]
    #                 if torch.is_tensor(sample_id):
    #                     sample_id = str(sample_id.item())
    #                 else:
    #                     sample_id = str(sample_id)

    #                 sample_dir = os.path.join(layer_dir, sample_id)
    #                 os.makedirs(sample_dir, exist_ok=True)

    #                 for head_idx in range(attn_tensor.size(1)): 
    #                     attn_head = attn_tensor[i, head_idx].detach().cpu().numpy()  # (L, L)

    #                  
    #                     save_path_npy = os.path.join(sample_dir, f"head{head_idx}.npy")
    #                     np.save(save_path_npy, attn_head)

    #                     
    #                     plt.figure()
    #                     plt.imshow(attn_head, cmap="viridis")
    #                     plt.colorbar()
    #                     plt.title(f"Layer {layer_idx} Sample {sample_id} Head {head_idx}")
    #                     save_path_png = os.path.join(sample_dir, f"head{head_idx}.png")
    #                     plt.savefig(save_path_png, dpi=150, bbox_inches='tight')
    #                     plt.close()

    return np.concatenate(probs), ids, np.array(labels_all)

def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # attn_save_dir = "/data/uestc/sunmingze/SMARTM6A/SeqStructMulAtn"
    # os.makedirs(attn_save_dir, exist_ok=True)
  
    df_test = pd.read_json(args.test, lines=True)
    seqs    = df_test['sequence'].to_list()
    ids_arr = np.asarray(df_test['id'].to_list())
    has_labels = 'label' in df_test.columns
    dummy_labels = df_test['label'].values.astype(np.float32) if has_labels else np.zeros(len(seqs), dtype=np.float32)
    dummy_ew     = np.ones(len(seqs), dtype=np.float32)

    test_ds = RNADataset(seqs, dummy_labels, ids_arr,
                         dummy_ew, args.path, training=False, k=args.kmers[0])
    loader  = DataLoader(test_ds, batch_size=args.batch, shuffle=False)

 
    all_pred = []
    for ckpt in args.weights:
        model = SMARTM6A(args.ntoken, args.nclass, args.ninp, args.nhead,
                             args.nhid, args.nlayers, True,
                             kmers=args.kmers, stride=args.stride,
                             dropout=args.dropout, finetune=True,return_aw=True).to(device)
        sd = torch.load(ckpt, map_location=device)
        model.load_state_dict(sd, strict=False)
        prob, ids, labels = predict_one_model(model, loader, device)
        all_pred.append(prob)

    mean_prob = np.mean(np.stack(all_pred, axis=0), axis=0)   # (N,)

     
    sub = pd.DataFrame({
        'id': ids,
        'prob': mean_prob,
        'true_label': labels  
    })
    sub.to_csv(args.out, index=False)
    print(f' Saved submission to {args.out}')

 
    if has_labels:
        y_true = labels
        y_pred = (mean_prob > 0.5).astype(int)
        print(f' ACC:  {accuracy_score(y_true, y_pred):.4f}')
        print(f' F1:   {f1_score(y_true, y_pred):.4f}')
        print(f' AUC:  {roc_auc_score(y_true, mean_prob):.4f}')
        print(f' PREC: {precision_score(y_true, y_pred):.4f}')
        print(f' RECALL:{recall_score(y_true, y_pred):.4f}')

if __name__ == '__main__':
    main()
