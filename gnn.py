import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
from model import *
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
import os
import time
import multiprocessing
cpu_cont = multiprocessing.cpu_count()
from contract_format_DFG import SourceFormatterAndDFG

vocab = SourceFormatterAndDFG("SolidityLexer.g4")
vocab.read_input_file()
vocab.remove_comments()
vocab = vocab.source_code
# Use regular expression to extract values in quotes
quoted_values = re.findall(r"'([^']*)'", vocab)
filtered_list = [element for element in quoted_values if '\n' not in element]
unique_list = list(set(filtered_list))

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

max_grad_norm = 1.0
train_batch_size = 32
per_gpu_train_batch_size=train_batch_size//1
local_rank = -1
epoch = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weight_decay = 0.0
learning_rate = 5e-5
adam_epsilon = 1e-8
n_gpu = torch.cuda.device_count()
output_dir = "model"
gradient_accumulation_steps = 1
start_step = 0
data = pd.read_csv("dataset.csv")
data = data.dropna()
sources = data["Text"].tolist()
labels = data["Label"].astype(int).tolist()

x_train, x_test, y_train, y_test = train_test_split(sources, labels, test_size= 0.2, random_state= 42)

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

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label)

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def evaluate(model, tokenizer, eval_when_training=False):
    # Loop to handle MNLI double evaluation (matched, mis-matched)

    eval_dataset = TextDataset(x_test, y_test, tokenizer)

    eval_batch_size = 32
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if local_rank == -1 else DistributedSampler(eval_dataset)
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
        inputs = batch[0].to(device)        
        label=batch[1].to(device) 
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
    train_batch_size = per_gpu_train_batch_size * 1
    train_sampler = RandomSampler(train_dataset) if local_rank == -1 else DistributedSampler(train_dataset)
    
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, 
                                  batch_size=train_batch_size,num_workers=4,pin_memory=True)
    max_steps= epoch*len( train_dataloader)
    save_steps=len( train_dataloader)
    warmup_steps=len( train_dataloader)
    logging_steps=len( train_dataloader)
    num_train_epochs=epoch
    model.to(device)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max_steps*0.1,
                                                num_training_steps=max_steps)

    # multi-gpu training (should be after apex fp16 initialization)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                          output_device=local_rank,
                                                          find_unused_parameters=True)

    # Train!
    print("***** Running training *****")
    print("  Num examples = %d", len(train_dataset))
    print("  Num Epochs = %d", num_train_epochs)
    print("  Instantaneous batch size per GPU = %d", per_gpu_train_batch_size)
    print("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                train_batch_size * gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if local_rank != -1 else 1))
    print("  Gradient Accumulation steps = %d", gradient_accumulation_steps)
    print("  Total optimization steps = %d", max_steps)
    
    global_step = start_step
    tr_loss, logging_loss,avg_loss,tr_nb,tr_num,train_loss = 0.0, 0.0,0.0,0,0,0
    best_mrr=0.0
    best_acc=0.0
    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()
 
    for idx in range(int(num_train_epochs)): 
        print(f"Epoch {idx+1}/{num_train_epochs}")
        # bar = tqdm(train_dataloader,total=len(train_dataloader))
        tr_num=0
        train_loss=0
        # for step, batch in enumerate(bar):
        for step, batch in tqdm(enumerate(train_dataloader)):
            inputs = batch[0].to(device)
            labels=batch[1].to(device) 
            model.train()
            loss,logits = model(inputs,labels)


            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps


            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            tr_loss += loss.item()
            tr_num+=1
            train_loss+=loss.item()
            if avg_loss==0:
                avg_loss=tr_loss
            avg_loss=round(train_loss/tr_num,5)

            # bar.set_description("epoch {} loss {}".format(idx, avg_loss))
            # logger.info("epoch {} loss {}".format(idx, avg_loss))

                
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
                    
                    if local_rank == -1:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(model, tokenizer,eval_when_training=True)
                        for key, value in results.items():
                            print("  %s = %s", key, round(value,4))                    
                        # Save model checkpoint
                        
                    if results['eval_acc']>best_acc:
                        best_acc=results['eval_acc']
                        print("  "+"*"*20)  
                        print("  Best acc:%s",round(best_acc,4))
                        print("  "+"*"*20)                          
                        
                        output_dir = "model.bin"
                        torch.save(model_to_save.state_dict(), output_dir)
                        print("Saving model checkpoint to %s", output_dir)
        avg_loss = round(train_loss / tr_num, 5)
        print("epoch {} loss {}".format(idx, avg_loss))

set_seed(42)

config= BertConfig.from_pretrained("bert-base-uncased")
config.num_labels = 1
model = BertForMaskedLM(config)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokenizer.add_tokens(unique_list)
model.resize_token_embeddings(len(tokenizer))

block_size = tokenizer.max_len_single_sentence
model = GNNReGVD(model, config, tokenizer)

print("Training/evaluation parameters")
train_dataset = TextDataset(x_train, y_train, tokenizer)
train(train_dataset, model, tokenizer)

results = {}
model.load_state_dict(torch.load('model.bin'))      
model.to(device)
result=evaluate(model, tokenizer)
print("***** Eval results *****")
for key in sorted(result.keys()):
    print("  %s = %s", key, str(round(result[key],4)))