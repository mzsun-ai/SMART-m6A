import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.nn.parameter import Parameter

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x *( torch.tanh(F.softplus(x)))
    
def gem(x, p=3, eps=1e-6):
    return F.avg_pool1d(x.clamp(min=eps).pow(p), (x.size(-1))).pow(1./p)

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class ScaledDotProductAttention(nn.Module):

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.gamma=nn.Parameter(torch.tensor(100.0))

    def forward(self, q, k, v, mask=None, attn_mask=None):

        attn = torch.matmul(q, k.transpose(2, 3))/ self.temperature
        to_plot=attn[0,0].detach().cpu().numpy()

        if mask is not None:
            attn = attn+mask*self.gamma
        if attn_mask is not None:

            attn=attn.float().masked_fill(attn_mask == 0, float('-1e-9'))

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn
    
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)
        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None,src_mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask  

        if src_mask is not None:
            src_mask=src_mask[:,:q.shape[2]].unsqueeze(-1).float()
            attn_mask=torch.matmul(src_mask,src_mask.permute(0,2,1))
            attn_mask=attn_mask.unsqueeze(1)
            q, attn = self.attention(q, k, v, mask=mask,attn_mask=attn_mask)
        else:
            q, attn = self.attention(q, k, v, mask=mask)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        return q, attn

class StructureTransformer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, k = 3):
        super(StructureTransformer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, d_model//nhead, d_model//nhead, dropout=dropout)
        self.mask_conv1 = nn.Sequential(nn.Conv2d(nhead//4,nhead//4,k),
                                        nn.BatchNorm2d(nhead//4),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(nhead//4,nhead,1),
                                        )
        self.mask_deconv = nn.Sequential(nn.ConvTranspose2d(nhead,nhead//4,k),
                                        nn.BatchNorm2d(nhead//4),
                                        nn.Sigmoid()
                                        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)


        self.activation = nn.ReLU()

        self.conv=nn.Conv1d(d_model,d_model,k,padding=0)
        self.deconv=nn.ConvTranspose1d(d_model,d_model,k)

    def forward(self, src , mask, src_mask=None):
        res = src
        src = self.norm3(self.conv(src.permute(0,2,1)).permute(0,2,1))
        mask_res=mask
        mask = self.mask_conv1(mask)
        src2,attention_weights = self.self_attn(src, src, src, mask=mask, src_mask=src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        src = res + self.dropout3(self.deconv(src.permute(0,2,1)).permute(0,2,1))
        src = self.norm4(src)
        mask = self.mask_deconv(mask)+mask_res
        return src,attention_weights,mask

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=200):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class SMARTM6A(nn.Module):

    def __init__(self, ntoken, nclass, ninp, nhead, nhid, nlayers, kmer_aggregation, kmers, stride=1,dropout=0.5,pretrain=False,return_aw=False,finetune=False):
        super(SMARTM6A, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        self.kmers=kmers
        self.transformer_encoder = []
        for i in range(nlayers):
            self.transformer_encoder.append(StructureTransformer(ninp, nhead, nhid, dropout, k=kmers[0]-i))
        self.transformer_encoder= nn.ModuleList(self.transformer_encoder)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.projection = nn.Linear(ninp*3, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp,nclass)
        self.mask_dense=nn.Conv2d(4,nhead//4,1)

        self.return_aw=return_aw
        self.pretrain=pretrain
        self.finetune = finetune
        if self.finetune:
                self.pooling = nn.AdaptiveMaxPool1d(1)
                self.classifier = nn.Sequential(
                nn.Linear(ninp, ninp//2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(ninp//2, nclass) )

        self.pretrain_decoders=nn.ModuleList()
        self.pretrain_decoders.append(nn.Linear(ninp,4))
        self.pretrain_decoders.append(nn.Linear(ninp,3))
        self.pretrain_decoders.append(nn.Linear(ninp,7))

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, mask=None,src_mask=None,return_encoded=False):
        B,L,_=src.shape
        src = src
        src = self.encoder(src).reshape(B,L,-1)
        src =self.projection(src)
        src = self.pos_encoder(src.permute(1,0,2)).permute(1,0,2)
        attention_weights = []
        mask=self.mask_dense(mask)
        for i,layer in enumerate(self.transformer_encoder):

            if src_mask is not None:
                src,attention_weights_layer,mask=layer(src, mask, src_mask[:,i])
            else:
                src,attention_weights_layer,mask=layer(src, mask)
            # attention_weights.append(attention_weights_layer.detach().cpu())
        # attention_weights=torch.stack(attention_weights).permute(1,0,2,3)
        encoder_output = src  # (B, L, ninp)
        if self.pretrain:
            ae_outputs = [decoder(encoder_output) for decoder in self.pretrain_decoders]
            return ae_outputs
        elif self.finetune:
            # attn_out, attn_weights_cls = self.multihead_attn(encoder_output, encoder_output, encoder_output)  # (B, L, ninp)
            pooled = self.pooling(encoder_output.transpose(1, 2)).squeeze(-1)  # (B, ninp)
            output = self.classifier(pooled)  # (B, nclass)

            if self.return_aw:
                # attn_mean_per_layer = [attn.mean(dim=1) for attn in attention_weights] 
                return output, {
                    "encoder":  attention_weights,
                    "classifier": None 
                }
            else:
                return output

        else:
            output = self.classifier(encoder_output)  # (B, L, nclass)
            if self.return_aw:
                attn_mean_per_layer = [attn.mean(dim=1) for attn in attention_weights] 
                return output, {
                    "encoder": attention_weights,
                    "classifier": None  #  attention
                }
            else:
                return output

