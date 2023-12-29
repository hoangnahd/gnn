import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from transformers import BertConfig, BertForMaskedLM, BertTokenizer, WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup
from model import *
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 label,

    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.label=label


def convert_examples_to_features(x, y, tokenizer):
    #source
    code=' '.join(x.split())
    code_tokens=tokenizer.tokenize(code)[:tokenizer.max_len_single_sentence-2]
    source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = tokenizer.max_len_single_sentence - len(source_ids)
    source_ids+=[tokenizer.pad_token_id]*padding_length
    return InputFeatures(source_tokens,source_ids,y)

class TextDataset(Dataset):
    def __init__(self, source, target, tokenizer):
        self.examples = []
        for x, y in zip(source, target):
            self.examples.append(convert_examples_to_features(x, y, tokenizer))

        for idx, example in enumerate(self.examples[:3]):
                print("*** Sample ***")
                print("Total sample".format(idx))
                print("idx: {}".format(idx))
                print("label: {}".format(example.label))
                print("input_tokens: {}".format([x.replace('\u0120','_') for x in example.input_tokens]))
                print("input_ids: {}".format(' '.join(map(str, example.input_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label)
    
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
config = BertConfig.from_pretrained("bert-base-uncased")
model = BertForMaskedLM(config)

data = pd.read_csv("dataset.csv")
data = data.dropna()
x = data["Text"].tolist()
y = data["Label"].astype(int).tolist()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=20, random_state=42)

model = GNNReGVD(model, config, tokenizer)

train_data = TextDataset(x_train, y_train, tokenizer)

num_classes = 2
max_length = 128
batch_size = 16
num_epochs = 4
learning_rate = 2e-5
train_batch_size = 32

train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size= train_batch_size,num_workers=4,pin_memory=True)

def train(model, data_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()  # You may need to adjust the loss function based on your task

    for batch in tqdm(data_loader, desc="Training"):
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs, labels=labels)

        # Calculate loss
        loss = criterion(outputs[0], labels)  # Assuming outputs[0] is the logits

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    average_loss = total_loss / len(data_loader)
    return average_loss

def predict_sentiment(text, model, tokenizer, device, max_length=128):
    model.eval()
    encoding = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
      outputs = model(input_ids=input_ids, attention_mask=attention_mask)
      _, preds = torch.max(outputs, dim=1)
    return "positive" if preds.item() == 1 else "negative"

def evaluate(model, data_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    return accuracy_score(actual_labels, predictions), classification_report(actual_labels, predictions)

train_batch_size = 32
eval_batch_size = 32
epoch = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
per_gpu_train_batch_size=train_batch_size//max(n_gpu,1)
per_gpu_eval_batch_size=eval_batch_size//max(n_gpu,1)
learning_rate = 5e-5
adam_epsilon = 1e-8
local_rank = -1
gradient_accumulation_steps = 1
fp16 = True
logging_steps = 50
fp16_opt_level = "O1"
import os

def evaluate(args, model, tokenizer,eval_when_training=False):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = "model"

    eval_dataset = TextDataset(x_test, y_test, tokenizer)

    if not os.path.exists(eval_output_dir) and local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    eval_batch_size = per_gpu_eval_batch_size * max(1, n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size,num_workers=4,pin_memory=True)

    # multi-gpu evaluate
    if n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    print("***** Running evaluation *****")
    print("  Num examples = %d", len(eval_dataset))
    print("  Batch size = %d", eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits=[]
    labels=[]
    for batch in eval_dataloader:
        inputs = batch[0].to(args.device)
        label=batch[1].to(args.device)
        with torch.no_grad():
            lm_loss,logit = model(inputs,label)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())
        nb_eval_steps += 1
    logits=np.concatenate(logits,0)
    labels=np.concatenate(labels,0)
    preds=logits[:,0]>0.5
    eval_acc=np.mean(labels==preds)
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)

    result = {
        "eval_loss": float(perplexity),
        "eval_acc":round(eval_acc,4),
    }
    return result

def train(train_dataset, model, tokenizer):
    """ Train the model """
    train_batch_size = per_gpu_train_batch_size * max(1, n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=train_batch_size,num_workers=4,pin_memory=True)
    max_steps=epoch*len( train_dataloader)
    save_steps=len( train_dataloader)
    warmup_steps=len( train_dataloader)
    logging_steps=len( train_dataloader)
    num_train_epochs=epoch
    model.to(device)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max_steps*0.1,
                                                num_training_steps=max_steps)
    try:
      from apex import amp
    except ImportError:
      raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    if local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                          output_device=local_rank,
                                                          find_unused_parameters=True)
    checkpoint_last = os.path.join("", 'checkpoint-last')
    scheduler_last = os.path.join("", 'scheduler.pt')
    optimizer_last = os.path.join("", 'optimizer.pt')
    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(torch.load(scheduler_last))
    if os.path.exists(optimizer_last):
        optimizer.load_state_dict(torch.load(optimizer_last))
    global_step = 0
    tr_loss, logging_loss,avg_loss,tr_nb,tr_num,train_loss = 0.0, 0.0,0.0,0,0,0
    best_mrr=0.0
    best_acc=0.0
    model.zero_grad()
    for idx in range(epoch):
      tr_num=0
      train_loss=0
      for step, batch in enumerate(train_dataloader):
            inputs = batch[0].to(device)
            labels=batch[1].to(device)
            model.train()
            loss,logits = model(inputs,labels)
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            if fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1.0)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            tr_loss += loss.item()
            tr_num+=1
            train_loss+=loss.item()
            if avg_loss==0:
                avg_loss=tr_loss
            avg_loss=round(train_loss/tr_num,5)
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                output_flag=True
                avg_loss=round(np.exp((tr_loss - logging_loss) /(global_step- tr_nb)),4)
                if local_rank in [-1, 0] and logging_steps > 0 and global_step % logging_steps == 0:
                    logging_loss = tr_loss
                    tr_nb=global_step
                if local_rank in [-1, 0] and save_steps > 0 and global_step % save_steps == 0:

                    if local_rank == -1 and evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(model, tokenizer,eval_when_training=True)
                        for key, value in results.items():
                            print("  %s = %s", key, round(value,4))
                        # Save model checkpoint

                    if results['eval_acc']>best_acc:
                        best_acc=results['eval_acc']
                        print("  "+"*"*20)
                        print("  Best acc:%s",round(best_acc,4))
                        print("  "+"*"*20)

                        checkpoint_prefix = 'checkpoint-best-acc'
                        output_dir = os.path.join("model", '{}'.format(checkpoint_prefix))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model,'module') else model
                        output_dir = os.path.join(output_dir, '{}'.format('model.bin'))
                        torch.save(model_to_save.state_dict(), output_dir)
                        print("Saving model checkpoint to %s", output_dir)
      avg_loss = round(train_loss / tr_num, 5)
      print("epoch {} loss {}".format(idx, avg_loss))

train(train_data, model, tokenizer)
