
import random
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from datasets import load_dataset
from transformers import AutoTokenizer

text_data = load_dataset("allenai/common_gen", split="train")
for i in range(4):
    print(text_data[i])
text_loader = torch.utils.data.DataLoader(text_data, batch_size=128, shuffle=True, num_workers=0)
try:
    for batch in text_loader:
        print(batch)
        break
except Exception as e:
    print("error:", e)

model_name = "openai/clip-vit-base-patch32"
tokenizer = AutoTokenizer.from_pretrained(model_name)

#把list转换为string
def add_eos_to_examples(example):
    string = ",".join(example['concepts'])  # "ski,mountain,skier"
    example['input_text'] = '%s .' % string
    example['target_text'] = '%s ' % example['target']
    return example


def convert_to_features(example_batch):
    input_encodings = tokenizer(example_batch['input_text'], padding="max_length", max_length=16, truncation=True,
                                return_tensors="pt")#因为句子长短不一，所以在句末bu加上padding，max_length=16表示句子长度不超过16
    target_encodings = tokenizer(example_batch['target_text'], padding="max_length", max_length=16, truncation=True,
                                 return_tensors="pt").input_ids
    labels_with_ignore_index = []
    for labels_example in target_encodings:
        labels_example = [label if label != 0 else -100 for label in labels_example]
        labels_with_ignore_index.append(labels_example)

    encodings = {
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'labels': labels_with_ignore_index
    }
    # print(encodings['input_ids'])

    return encodings
def collate(batch):
    input_ids = [torch.tensor(item['input_ids']) for item in batch]
    attention_mask = [torch.tensor(item['attention_mask']) for item in batch]
    labels = [item['labels'] for item in batch]

    input_ids = torch.stack(input_ids)
    attention_mask = torch.stack(attention_mask)
    labels = torch.tensor(labels)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

text_data = text_data.map(add_eos_to_examples, batched=False, remove_columns=text_data.column_names)
# print(text_data[0])# { 'input_text': 'ski,mountain,skier .', 'target_text': 'skiing '}
print(text_data[0])
text_data = text_data.map(convert_to_features, batched=True, remove_columns=text_data.column_names)
print(text_data[0])
# 作业：把这句话删了，采用自己定义collate_fn的方法获得正确的输出

text_loader = torch.utils.data.DataLoader(text_data, batch_size=4, shuffle=False,collate_fn=collate, num_workers=0)
try:
    for batch in text_loader:
        print(batch)
        break
except Exception as e:
    print(e)
