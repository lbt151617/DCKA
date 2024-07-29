import torch.nn as nn
import torch
from load_data import load_kd
import torch.nn.functional as F
import torch.nn.init as init  
from transformers import PreTrainedModel
import math
from loss_fn import FocalLoss
from transformers import BertPreTrainedModel, BertModel, BertTokenizer
kd_feature = load_kd()


    
class MultiClassFocalLossWithAlpha(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        """
        :param alpha: 权重系数列表，三分类中第0类权重0.2，第1类权重0.3，第2类权重0.5
        :param gamma: 困难样本挖掘的gamma
        :param reduction:
        """
        super(MultiClassFocalLossWithAlpha, self).__init__()
        self.alpha = torch.tensor(alpha).cuda()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        alpha = self.alpha[target]  # 为当前batch内的样本，逐个分配类别权重，shape=(bs), 一维向量
        log_softmax = torch.log_softmax(pred, dim=1) # 对模型裸输出做softmax再取log, shape=(bs, 3)
        logpt = torch.gather(log_softmax, dim=1, index=target.view(-1, 1))  # 取出每个样本在类别标签位置的log_softmax值, shape=(bs, 1)
        logpt = logpt.view(-1)  # 降维，shape=(bs)
        ce_loss = -logpt  # 对log_softmax再取负，就是交叉熵了
        pt = torch.exp(logpt)  #对log_softmax取exp，把log消了，就是每个样本在类别标签位置的softmax值了，shape=(bs)
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss  # 根据公式计算focal loss，得到每个样本的loss值，shape=(bs)
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss
        

        
class TextCNN(nn.Module):
    def __init__(self):
        super().__init__()     
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=256, kernel_size=(fs, 1024)) 
            for fs in (2, 3, 4, 5)
        ])
        
        
    def forward(self, emb):
        emb = emb.unsqueeze(1)
        conved = [F.relu(conv(emb)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        pooled = torch.cat(pooled, 1)
        return pooled
        
class BiLSTM(nn.Module):
    def __init__(self):
        super().__init__()    
        self.lstm=nn.LSTM(1024, 256, 1, bidirectional=True, batch_first=False)
        
    def forward(self, emb):
        emb = emb.permute(1, 0, 2)
        out, (h, c) = self.lstm(emb)
        out = torch.cat((out[0], out[-1]), -1)
        return out

class cross_attention(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()    
        self.q = nn.Linear(dim, dim, bias=True)
        self.k = nn.Linear(dim, dim, bias=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k):
        k = self.k(k)
        att = torch.matmul(q, k.T)/math.sqrt(q.size(-1))
        out = torch.matmul(att, k)
        return q + self.dropout(out)
        
class BertForTCMClassification(PreTrainedModel):  
    def __init__(self, config):  
        super(BertForTCMClassification, self).__init__(config) 
        self.num_labels = config.num_labels
        self.bert = BertModel.from_pretrained('zy-bert')
        self.config = config
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.cnn = TextCNN()
        self.rnn = BiLSTM()
        self.ch_cross_att = cross_attention(config.hidden_size, 0.5)
        self.de_cross_att = cross_attention(config.hidden_size, 0.5)
        self.num_labels = config.num_labels
        self.kd_feature = kd_feature
        self.w_k = nn.Linear(config.hidden_size, config.hidden_size)
        self.init_weights()
        
    def forward(self, ch_input_ids=None, ch_attention_mask=None, ch_token_type_ids=None,
                de_input_ids=None, de_attention_mask=None, de_token_type_ids=None, labels=None):
        # 调用ERNIE模型进行前向传播，传入输入ID和token类型ID
        ch_encoding = self.bert(input_ids=ch_input_ids, token_type_ids=ch_token_type_ids, attention_mask=ch_attention_mask)
        de_encoding = self.bert(input_ids=de_input_ids, token_type_ids=de_token_type_ids, attention_mask=de_attention_mask)
        # 取出第二个输出，即池化后的输出
        # pooled_output = outputs[1]
        ch_h = ch_encoding.last_hidden_state
        de_h = de_encoding.last_hidden_state
        
        ch_h = self.cnn(ch_h)
        de_h = self.rnn(de_h)
        
        # de = self.de_cross_att(de_h, self.kd_feature[1])
        # de = self.de_cross_att(de_h, self.kd_feature)  

        ch_de = ch_h + de_h
        ch_de = self.ch_cross_att(ch_de, self.kd_feature)
        # ch_de = torch.cat((ch, de), -1)
        pooled_output = self.dropout(ch_de)

        # 将池化后的输出传入分类器，得到logits
        logits = self.classifier(pooled_output)
        # 将logits赋值给outputs
        outputs = logits
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # al = [1.0] * 148
            # loss_fct = MultiClassFocalLossWithAlpha(alpha=al).cuda()
            # loss_fct = FocalLoss().cuda()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return {
                'loss':loss,
                'logits':logits
            }
        return {"logits": logits} 