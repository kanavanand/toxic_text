import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"



MAX_LEN = 512
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 1e-05
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', truncation=True, do_lower_case=True)

class MultiLabelDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len, new_data=False):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.comment_text
        self.new_data = new_data
        
        if not new_data:
            self.targets = self.data.labels
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        out = {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
        }
        
        if not self.new_data:
            out['targets'] = torch.tensor(self.targets[index], dtype=torch.float)

        return out

class DistilBERTClass(torch.nn.Module):
    def __init__(self):
        super(DistilBERTClass, self).__init__()
        
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(768, 768),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(768, 6)
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        out = hidden_state[:, 0]
        out = self.classifier(out)
        return out

model = DistilBERTClass()
model.to(DEVICE);

model_loaded = torch.load('model/inference_models_output_4fold_distilbert_fold_best_model.pth',map_location=torch.device('cpu'))

model.load_state_dict(model_loaded['model'])


val_params = {'batch_size': VALID_BATCH_SIZE,
               'shuffle': False,
    
                }
def give_toxic(text):
    text = "You fucker "
    test_data = pd.DataFrame([text],columns=['comment_text'])
    test_set = MultiLabelDataset(test_data, tokenizer, MAX_LEN, new_data=True)
    test_loader = DataLoader(test_set, **val_params)

    all_test_pred = []

    def test(epoch):
        model.eval()

        with torch.inference_mode():

            for _, data in tqdm(enumerate(test_loader, 0)):


                ids = data['ids'].to(DEVICE, dtype=torch.long)
                mask = data['mask'].to(DEVICE, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(DEVICE, dtype=torch.long)
                outputs = model(ids, mask, token_type_ids)
                probas = torch.sigmoid(outputs)

                all_test_pred.append(probas)

    probas = test(model)

    all_test_pred = torch.cat(all_test_pred)

    label_columns = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    preds = all_test_pred.detach().cpu().numpy()[0]

    final_dict  = dict(zip(label_columns , preds))
    return final_dict

def device():
    return DEVICE

print(give_toxic("fuck"))