---
layout: post
title: Aspect Based Sentiment Analysis
date: 2023-07-06 15:09:00
description: Developed a system to extract the aspect terms and jointly detect the polarity of these terms
tags: PyTorch, PyTorch-Lightning, nlp, sentiment-analysis, aspect-extraction, polarity-detection
categories: nlp 
featured: true
---

The objective of this activity is to develop a system to extract the aspect terms and jointly detect the polarity. This is useful for the companies to understand the customer feedback and improve their products and services. For example, consider the following sentence:

> The rope is strong but the handle is weak.
>
> **Aspect terms:** rope, handle
> **Polarity:** positive, negative
> **Polarity of sentence:** neutral


The massive amount of data available on the internet makes it difficult to manually classify the feedback. The objective of this activity is to develop a system to extract the aspect terms and jointly detect the polarity of these terms. This would speed up the decision making process for the companies. Also, these systems can be used in a more complex pipeline to detect trends in the feedback and help the companies improve their products and services.

The dataset used to train the model has been the [SemEval2014](https://github.com/lixin4ever/E2E-TBSA/tree/master/data). To extract the sentences in the proper format you can use the utils in their repository. Also, we could decide to use their splitting of the data or discard it and use our own. In this case, we collect the data and split it by ourselves.

We can dowload the repository as follows:

```bash
git clone https://github.com/lixin4ever/E2E-TBSA.git
```

#### Load and preprocess the data

We import the libraries we will use:


```python
from transformers import RwkvModel

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
import pytorch_lightning as pl

from sklearn.model_selection import StratifiedKFold, KFold
from transformers import AutoTokenizer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pickle
import glob

import E2E-TBSA.module.utils as utils # This is the utils from the repository

# Set the device to cuda if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Additionally, we define some functions to preprocess the data (such as padding the sentences, etc.)

```python
# merge two lists
def merge_lists(a, b):
    return a+b

def pad(dataset, max_len = 464):
    for i in range(len(dataset)):
        dataset[i]['words'] = dataset[i]['words'] + ['PAD'] * (max_len - len(dataset[i]['words']))
        dataset[i]['ts_raw_tags'] = dataset[i]['ts_raw_tags'] + ['PAD'] * (max_len - len(dataset[i]['ts_raw_tags']))
        dataset[i]['ote_raw_tags'] = dataset[i]['ote_raw_tags'] + ['PAD'] * (max_len - len(dataset[i]['ote_raw_tags']))
    return dataset

def retrieve_the_sentences_word_padded(d):
    return [' '.join(d_words['words']) for d_words in d]


def retrieve_the_sentences(d):
    return [d_s['sentence'] for d_s in d]
```

Then, to preprocess the data you can do the following:

```python
def load_save_the_data():
    data_path = glob.glob('./E2E-TBSA/module/data/*.txt')

    dataset = []
    for path in data_path:
        temp = utils.read_data(path)
        dataset = merge_lists(dataset, temp)

    print('Number of sample: ', len(dataset))
    print('Length max sentence: ', max([len(d['sentence']) for d in dataset]))
    print('Length min sentence: ', min([len(d['sentence']) for d in dataset]))

    print(dataset[0].keys())
    print(dataset[2]['ts_raw_tags'], dataset[44]['ote_raw_tags'])

    dict_map_aspect = {'O': 0, 'T-NEG': 1, 'T-POS': 2, 'T-NEU': 3, 'PAD': 4}
    dict_map_aspect_classify = {'O': 0, 'T': 1, 'PAD': 2}

    print(dataset[0]['sentence'])

    dataset_padded = pad(dataset)

    label_aspect_classification = torch.stack([
        torch.stack([torch.tensor(dict_map_aspect_classify[tag]) for tag in dataset_padded[i]['ote_raw_tags']]) for i in range(len(dataset_padded))
    ])

    label_aspect = torch.stack([
        torch.stack([torch.tensor(dict_map_aspect[tag]) for tag in dataset_padded[i]['ts_raw_tags']]) for i in range(len(dataset_padded))
    ])

    print('ASPECT classification: ',label_aspect_classification.shape, 'ASPECT sentiment: ',label_aspect.shape)

    torch.save(label_aspect_classification, './dataset/label_aspect_classification.pt')
    torch.save(label_aspect, './dataset/label_aspect.pt')  
    with open('./dataset/dataset_padded.pkl', 'wb') as f:
        pickle.dump(dataset_padded, f, pickle.HIGHEST_PROTOCOL)

    sentences = retrieve_the_sentences_word_padded(dataset_padded)
    tokenizer = AutoTokenizer.from_pretrained("RWKV/rwkv-4-169m-pile")
    tokenizer.pad_token = 'PAD'
    inputs = tokenizer(sentences, return_tensors="pt", padding=True)
    print('Input: ', inputs['input_ids'].shape)

    torch.save(inputs["input_ids"], "./dataset/input_ids.pt")
    torch.save(inputs["attention_mask"], "./dataset/attention_mask.pt")

load_save_the_data() # load and save the data
```

#### Architecture

Here we define the backbone of our architecture. We use the pretrained model from RWKV. 

```python
class RWKV_Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained = RwkvModel.from_pretrained("RWKV/rwkv-4-169m-pile")

        # # Freeze the pretrained model
        # for param in self.pretrained.parameters():
        #     param.requires_grad = False

    def forward(self, input_ids):
        outputs = self.pretrained(input_ids)
        return outputs.last_hidden_state
```

The model relies on the features extracted by the backbone (RWKV model). Then, we use an MLP with non-linearity, dropout and layer normalization to classify the aspect terms. We also use a mask to discard the non-aspect terms. In this way, the subsequent module has to focus only on the aspect terms. This simplifies the overall effort of the next module to classify the polarity of the aspect terms.

```python

def non_aspect_masking(out, labels):
    # print(out.shape, labels.unsqueeze(-1).shape)
    return torch.mul(out, (labels.unsqueeze(-1) == 1).reshape(out.shape[0], 960, 1).repeat(1, 1, 768))

class AspectExtraction(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = RWKV_Backbone()
        self.aspect_classfy = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            nn.Linear(256, 3),
        )

        self.mask_non_aspect = non_aspect_masking

    def forward(self, input_ids):

        # Get the output from the backbone
        out = self.backbone(input_ids)

        # print(output.shape)

        # Get the aspect classification
        aspect_classfy = self.aspect_classfy(out)

        # print(aspect_classfy.shape)

        # Get the logits
        logits = torch.argmax(aspect_classfy, dim=-1)

        # Mask the non-aspect words
        out = self.mask_non_aspect(out, logits)

        return out, aspect_classfy
```

This module classifies the polarity of the aspect terms and moves forward the aspect classified.

```python

class AspectClassification(nn.Module):
    def __init__(self):
        super().__init__()

        self.AspectExtraction = AspectExtraction()

        self.sentiment_classfy  = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            nn.Linear(256, 4),
        )

    def forward(self, input_ids):

        # Get the the masked output
        output, aspect_classfy = self.AspectExtraction(input_ids)

        # Get the sentiment classification
        sentiment_classfy = self.sentiment_classfy(output)

        return sentiment_classfy, aspect_classfy
```

Finally, we define the training loop. We use PyTorch-Lightning to simplify the training loop. We use the AdamW optimizer with a cosine annealing scheduler with warm restarts. We use the cross entropy loss for both the aspect classification and the sentiment classification. We also use the accuracy and the F1 score as metrics. The advantage of using PyTorch-Lightning is that we can easily log the metrics and visualize them using TensorBoard. Also, we can easily save the model and load it later.

```python
class Net(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.model = AspectClassification()

        self.loss_sentiment_classfy = nn.CrossEntropyLoss()
        self.loss_aspect_classfy = nn.CrossEntropyLoss()

        self.acc_aspect_classfy = torchmetrics.Accuracy(task='multiclass', num_classes=3)
        self.f1_aspect_classfy = torchmetrics.F1Score(task='multiclass', num_classes=3)

        self.acc_sentiment_classfy = torchmetrics.Accuracy(task='multiclass', num_classes=5)
        self.f1_sentiment_classfy = torchmetrics.F1Score(task='multiclass', num_classes=5)

    def forward(self, input_ids):
        return self.model(input_ids)

    def training_step(self, batch, batch_idx):
        input_ids, labels_aspect_classificaiton, labels_sentiment_classificaiton = batch

        sentiment_classfy, aspect_classfy = self(input_ids)

        loss_aspect_classfy = self.loss_aspect_classfy(aspect_classfy, labels_aspect_classificaiton)
        loss_sentiment_classfy = self.loss_sentiment_classfy(sentiment_classfy, labels_sentiment_classificaiton)

        loss = loss_aspect_classfy + loss_sentiment_classfy

        self.log('train_loss', loss,    on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss_aspect_classfy', loss_aspect_classfy,   on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss_sentiment_classfy', loss_sentiment_classfy,   on_step=True, on_epoch=True, prog_bar=True, logger=True)

        labels_aspect_classificaiton = torch.argmax(labels_aspect_classificaiton, dim=-1)
        labels_sentiment_classificaiton = torch.argmax(labels_sentiment_classificaiton, dim=-1)
        sentiment_classfy = torch.argmax(sentiment_classfy, dim=-1)
        aspect_classfy = torch.argmax(aspect_classfy, dim=-1)

        acc_aspect_classfy = self.acc_aspect_classfy(aspect_classfy, labels_aspect_classificaiton)
        f1_aspect_classfy = self.f1_aspect_classfy(aspect_classfy, labels_aspect_classificaiton)

        acc_sentiment_classfy = self.acc_sentiment_classfy(sentiment_classfy, labels_sentiment_classificaiton)
        f1_sentiment_classfy = self.f1_sentiment_classfy(sentiment_classfy, labels_sentiment_classificaiton)

        self.log('train_acc_aspect_classfy', acc_aspect_classfy,   on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_f1_aspect_classfy', f1_aspect_classfy,   on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.log('train_acc_sentiment_classfy', acc_sentiment_classfy,   on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_f1_sentiment_classfy', f1_sentiment_classfy,   on_step=True, on_epoch=True, prog_bar=True, logger=True)


        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, labels_aspect_classificaiton, labels_sentiment_classificaiton = batch

        sentiment_classfy, aspect_classfy = self(input_ids)

        # print(sentiment_classfy.shape, aspect_classfy.shape)

        loss_aspect_classfy = self.loss_aspect_classfy(aspect_classfy, labels_aspect_classificaiton)
        loss_sentiment_classfy = self.loss_sentiment_classfy(sentiment_classfy, labels_sentiment_classificaiton)

        loss = loss_aspect_classfy + loss_sentiment_classfy

        self.log('val_loss', loss,    on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_loss_aspect_classfy', loss_aspect_classfy,   on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_loss_sentiment_classfy', loss_sentiment_classfy,   on_step=True, on_epoch=True, prog_bar=True, logger=True)

        labels_aspect_classificaiton = torch.argmax(labels_aspect_classificaiton, dim=-1)
        labels_sentiment_classificaiton = torch.argmax(labels_sentiment_classificaiton, dim=-1)
        sentiment_classfy = torch.argmax(sentiment_classfy, dim=-1)
        aspect_classfy = torch.argmax(aspect_classfy, dim=-1)

        acc_aspect_classfy = self.acc_aspect_classfy(aspect_classfy, labels_aspect_classificaiton)
        f1_aspect_classfy = self.f1_aspect_classfy(aspect_classfy, labels_aspect_classificaiton)

        acc_sentiment_classfy = self.acc_sentiment_classfy(sentiment_classfy, labels_sentiment_classificaiton)
        f1_sentiment_classfy = self.f1_sentiment_classfy(sentiment_classfy, labels_sentiment_classificaiton)

        self.log('val_acc_aspect_classfy', acc_aspect_classfy,   on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_f1_aspect_classfy', f1_aspect_classfy,   on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.log('val_acc_sentiment_classfy', acc_sentiment_classfy,   on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_f1_sentiment_classfy', f1_sentiment_classfy,   on_step=True, on_epoch=True, prog_bar=True, logger=True)


        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
        return [optimizer], [scheduler]
```

#### Training

Lastly, this function is used to get the train_dataloader and the val_dataloader given the k-fold split (We use as Batch Size 128).


```python
BATCH_SIZE = 128

def get_dataloader(input_ids, labels_aspect_classificaiton, labels_sentiment_classificaiton, train_idx, val_idx):

    temp_train_input_ids = input_ids[train_idx].to(device)
    temp_train_labels_aspect_classificaiton = labels_aspect_classificaiton[train_idx].to(device)
    temp_train_labels_sentiment_classificaiton = labels_sentiment_classificaiton[train_idx].to(device)

    temp_val_input_ids = input_ids[val_idx].to(device)
    temp_val_labels_aspect_classificaiton = labels_aspect_classificaiton[val_idx].to(device)
    temp_val_labels_sentiment_classificaiton = labels_sentiment_classificaiton[val_idx].to(device)

    # initialize the train dataset
    train_dataset = torch.utils.data.TensorDataset(temp_train_input_ids, temp_train_labels_aspect_classificaiton, temp_train_labels_sentiment_classificaiton)
    del temp_train_input_ids, temp_train_labels_aspect_classificaiton, temp_train_labels_sentiment_classificaiton
    # initialize the train dataloader
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # initialize the val dataset
    val_dataset = torch.utils.data.TensorDataset(temp_val_input_ids, temp_val_labels_aspect_classificaiton, temp_val_labels_sentiment_classificaiton)
    del temp_val_input_ids, temp_val_labels_aspect_classificaiton, temp_val_labels_sentiment_classificaiton
    # initialize the val dataloader
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    return train_dataloader, val_dataloader
```

To train the model with k-fold cross validation you can do the following:

```python
N_FOLDS = 5
N_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 2

# load the precomputed data
input_ids = torch.load("dataset/input_ids.pt")
labels_aspect_classificaiton = torch.load("dataset/label_aspect_classification.pt")
labels_sentiment_classificaiton = torch.load("dataset/label_aspect.pt")
print('Done!')
print()


print('Prepare and split the data...')
for idx in range(labels_sentiment_classificaiton.shape[0]):
    temp = torch.zeros(labels_sentiment_classificaiton[idx].size())
    temp[labels_sentiment_classificaiton[idx]==1] = 1
    temp[labels_sentiment_classificaiton[idx]==2] = 2
    temp[labels_sentiment_classificaiton[idx]==3] = 3
    labels_sentiment_classificaiton[idx] = temp

# pad the labels
labels_aspect_classificaiton = F.pad(labels_aspect_classificaiton, (0, 960-labels_aspect_classificaiton.shape[1]))
labels_sentiment_classificaiton = F.pad(labels_sentiment_classificaiton, (0, 960-labels_sentiment_classificaiton.shape[1]))

# one hot encoding
labels_aspect_classificaiton = torch.nn.functional.one_hot(labels_aspect_classificaiton.to(torch.int64), num_classes=3).to(torch.float32)
labels_sentiment_classificaiton = torch.nn.functional.one_hot(labels_sentiment_classificaiton.to(torch.int64), num_classes=4).to(torch.float32)

# initialize the kfold
kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
# use a random_state to ensure reproducibility

# initialize the model checkpoint
checkpoint_callback = ModelCheckpoint(
    monitor='val_acc_sentiment_classfy',
    dirpath='bin/sentiment_classfy/',
    filename='aspect_extraction_sentiment_classfy_test2-{epoch:02d}-{val_loss:.2f}',
    save_top_k=1,
    mode='max',
)

# initialize the early stopping
early_stop_callback = EarlyStopping(
    monitor='val_acc_sentiment_classfy',
    min_delta=0.00,
    patience=EARLY_STOPPING_PATIENCE,
    verbose=True,
    mode='max'
)
print('Done!')

print('Start the training...')

# loop over the kfold
for fold, (train_idx, val_idx) in enumerate(kfold.split(input_ids, labels_aspect_classificaiton)):

    train_dataloader, val_dataloader = get_dataloader(input_ids, labels_aspect_classificaiton, labels_sentiment_classificaiton, train_idx, val_idx)

    # initialize the model
    model = Net().to(device)

    # initialize the trainer
    trainer = pl.Trainer(
        accelerator='auto',
        max_epochs=N_EPOCHS,
        callbacks=[checkpoint_callback, early_stop_callback],
    )

    # train the model
    trainer.fit(model, train_dataloader, val_dataloader)

print('Done!')
```
*Note: The larger the number of folds, the longer the training time, on the other hand, the results are more reliable.*

Pythorch-Lightning will save the best model in the folder `bin/sentiment_classfy/`. You can load the model as follows:

```python
model = Net.load_from_checkpoint('bin/sentiment_classfy/aspect_extraction_sentiment_classfy_test-epoch=10-val_loss=0.00.ckpt')
```

Also, it logs the metrics in the folder `lightning_logs/`. You can visualize the metrics using TensorBoard in a jupyter-cell as follows:

```bash
tensorboard --logdir lightning_logs/
```

#### Results

After training the table reports the results obtained on the testset:

| Fold | lr  | AspectAccuracy | AspF1 |  AspPrecision | AspRecall | PolarityAccuracy | PolarityF1 | PolarityPrecision | PolarityRecall |
|------|-----|----------------|-------|---------------|-----------|------------------|------------|-------------------|----------------|
| 0    | 5e-5| 96.7%          | 93.9% | 92%         | 95.8%     | 96.8%            | 94.3%   | 99.2%             | 93.3%          |
| 3    | 5e-5| 96.4%          | 93% | 93%         | 95%     | 97%            | 94%   | 98.1%             | 93.1%          |

Due to its representativeness, a proper splitting of the dataset can lead the model to better generalize on the testset. Even if on the testset some models achieve better scores is not the case that in real-world scenarios the model will perform better. For example, the model that achieves the best score on the validation set might have a lower score on the testset. In fact, the validation set might not be representative of the testset. For this reason, it is important to use a proper splitting of the dataset to have a better estimation of the model performance.

#### Conclusion

In this activity, we developed a system to extract the aspect terms and jointly detect the polarity of these terms. Then we reported the results obtained from the test. The model can be the starting point for a more complex pipeline to detect trends in the feedback and help the companies improve their products and services.

---
References:

1. [SemEval2014](https://github.com/lixin4ever/E2E-TBSA/tree/master/data)

---