import json
from torch.utils.data import DataLoader, Dataset
import torch
import transformers
from transformers import BertTokenizer, BertModel
transformers.logging.set_verbosity_error()
class CustomDataset(Dataset):
    def __init__(self, des, ch, det, labels, tokenizer):
        self.ch = tokenizer(ch, truncation=True, padding='max_length', max_length=20, return_tensors='pt')
        self.de = tokenizer(des, det, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
        self.labels = labels
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        item = {'ch_input_ids': self.ch['input_ids'][idx], 'ch_attention_mask': self.ch['attention_mask'][idx], 'ch_token_type_ids': self.ch['token_type_ids'][idx]}
        item['labels'] = torch.tensor(self.labels[idx])
        item['de_input_ids'] = self.de['input_ids'][idx]
        item['de_attention_mask'] = self.de['attention_mask'][idx]
        item['de_token_type_ids'] = self.de['token_type_ids'][idx]
        return item

    def __len__(self):
        return len(self.labels)


class_name = open('data/syndrome_vocab.txt', 'r', encoding='utf-8').read().split('\n')
def get_data(txt_file_path, tokenizer):
    with open(txt_file_path, 'r', encoding='utf-8') as file:
        des, chi, det, labels = [], [], [], []
        for line in file:
            items = json.loads(line)
            des.append(items['description'])
            det.append(items['detection'])
            chi.append(items['chief_complaint'])
            labels.append(class_name.index(items['norm_syndrome']))
        file.close()
    
    return des, chi, det, labels

def process_data():
    tokenizer = BertTokenizer.from_pretrained('zy-bert')
    train_des, train_chi, train_det, train_labels = get_data('data/train.json', tokenizer)
    train_dataset = CustomDataset(train_des, train_chi, train_det, train_labels, tokenizer)
    test_des, test_chi, test_det, test_labels = get_data('data/test.json', tokenizer)
    test_dataset = CustomDataset(test_des, test_chi, test_det, test_labels, tokenizer)
    dev_des, dev_chi, dev_det, dev_labels = get_data('data/dev.json', tokenizer)
    dev_dataset = CustomDataset(dev_des, dev_chi, dev_det, dev_labels, tokenizer)
    return train_dataset, test_dataset, dev_dataset

def load_kd():
    tokenizer = BertTokenizer.from_pretrained('zy-bert')
    model = BertModel.from_pretrained('zy-bert')
    model = model.cuda()
    kds = open(r'data/knowledge.txt', 'r', encoding='utf-8').read().split('\n')
    de_corpus = []
    pe_corpus = []
    co_corpus = []
    for kd in kds:
        name, de, pe, co = kd.split('\t')
        de_corpus.append(de)
        pe_corpus.append(pe)
        co_corpus.append(co)
        
    encodings = tokenizer(de_corpus, pe_corpus, co_corpus, truncation=True, padding='max_length', max_length=512)
    input_ids =torch.tensor(encodings['input_ids']).cuda()
    token_ids = torch.tensor(encodings['token_type_ids']).cuda()
    attention_mask = torch.tensor(encodings['attention_mask']).cuda()

    # pe_encodings = tokenizer(pe_corpus, co_corpus, truncation=True, padding='max_length', max_length=512)
    # pe_input_ids =torch.tensor(pe_encodings['input_ids']).cuda()
    # pe_token_ids = torch.tensor(pe_encodings['token_type_ids']).cuda()
    # pe_attention_mask = torch.tensor(pe_encodings['attention_mask']).cuda()
    
    with torch.no_grad():
        de_feature = model(input_ids, token_type_ids=token_ids, attention_mask=attention_mask)
        # pe_feature = model(pe_input_ids, token_type_ids=pe_token_ids, attention_mask=pe_attention_mask)
    return de_feature[1]
    
    
    
    
    

