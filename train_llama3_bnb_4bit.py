# !pip install -U "transformers>=4.42.3" bitsandbytes accelerate peft

import os
import copy
from dataclasses import dataclass
from sklearn.model_selection import train_test_split  
from sympy import false
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from datasets import Dataset
from scipy.special import softmax
from sklearn.preprocessing import LabelEncoder
from transformers import (
    BitsAndBytesConfig,
    LlamaPreTrainedModel,
    LlamaForCausalLM,
    LlamaModel,
    AutoTokenizer,
    PreTrainedTokenizerBase, 
    EvalPrediction,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from sklearn.metrics import log_loss, accuracy_score
import torch, gc

# 这两行是清理内存的
gc.collect()
torch.cuda.empty_cache()

ADD_CSV = '/root/CODE/Datasets/llama3/ds_add_train.csv'
TRAIN_CSV = "/root/CODE/Datasets/llama3/ds_train_new.csv" # 训练集路径
VAL_CSV = "/root/CODE/Datasets/llama3/ds_val_new.csv" 
model_path = "/root/CODE/models/llama-3-8b-Instruct-bnb-4bit"# 模型路径
# WEIGHTS_PATH = "/root/output/Gemma2_bnb_4bit/checkpoint-6000"
drop_d = True # 是否去掉训练集中问题和标签重复的行
is_add_data = True # 是否增加add——train来训练

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LENGTH = 1024

target_columns = ['winner_model_a', 'winner_model_b', 'winner_tie']
columns_to_vectorize = ["prompt", "response_a", "response_b"]
train_data = pd.read_csv(TRAIN_CSV)

if drop_d: # 去掉问题和标签重复的行
    train_data = train_data.drop_duplicates(subset=['prompt', 'label'], keep='last')

if is_add_data:
    add_data = pd.read_csv(ADD_CSV)
    train_data = pd.concat([add_data, train_data], axis=0).reset_index(drop=True) # 先训add，再训train
    
print(f'训练集共有{len(train_data)}条')
val_data = pd.read_csv(VAL_CSV)


tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.add_eos_token = True
tokenizer.padding_side = 'right'

LABEL_IDS = [tokenizer(i, add_special_tokens=False)["input_ids"][0] for i in ['first', 'second', 'tie']]

def tokenize(example, tokenizer):
    prompt = tokenizer('question: ' + " ".join(eval(example['prompt'], {"null": ""})), add_special_tokens=False)["input_ids"]
    response_a = tokenizer('\n\n<first response>: ' + " ".join(eval(example['response_a'], {"null": ""})), add_special_tokens=False)["input_ids"]
    response_b = tokenizer('\n\n<second response>: ' + " ".join(eval(example['response_b'], {"null": ""})), add_special_tokens=False)["input_ids"]
    
    if len(prompt+response_a+response_b) > MAX_LENGTH:
        prompt = tokenizer('question: ' + eval(example['prompt'], {"null": ""})[-1], add_special_tokens=False)["input_ids"][:256]
        response_a = tokenizer('\n\n<first response>: ' + eval(example['response_a'], {"null": ""})[-1], add_special_tokens=False)["input_ids"][:512]
        response_b = tokenizer('\n\n<second response>: ' + eval(example['response_b'], {"null": ""})[-1], add_special_tokens=False)["input_ids"][:512]
    extra_prompt = tokenizer("\n----\n which is the better response? Choose 'first' if you believe <first response> is superior, choose 'second' if you think <second response> is better, choose 'tie' if you find both responses to be equally good or equally bad. \n\nAnswer: ", add_special_tokens=False)["input_ids"]

    label_token_id = LABEL_IDS[int(example['label'])]
    input_ids = [tokenizer.bos_token_id] + prompt + response_a + response_b + extra_prompt + [label_token_id] + [tokenizer.eos_token_id]
    attention_mask = len(input_ids)*[1]
    labels = [-100]* len([tokenizer.bos_token_id] + prompt + response_a + response_b + extra_prompt) + [label_token_id] + [tokenizer.eos_token_id]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


def load_data(df, tokenizer):
    raw_datasets = Dataset.from_pandas(df)
    tokenized_datasets = raw_datasets.map(
        tokenize, 
        remove_columns=raw_datasets.column_names,
        fn_kwargs={'tokenizer': tokenizer}
    )
    return tokenized_datasets

def compute_metrics(pred):
    logits, labels = pred
    preds = logits.argmax(axis=-1) # 取最大概率的token
    label_tokens_ids = np.array(LABEL_IDS)
    index_mapping = {value.item(): idx for idx, value in enumerate(label_tokens_ids)}
    labels = labels[np.isin(labels, label_tokens_ids)]
    labels = np.array([index_mapping[label.item()] for label in labels])
    acc = accuracy_score(labels, preds) # 算精度
    probs = softmax(logits, axis=-1)
    log_loss_ = log_loss(labels, probs) # 交叉熵
    return {'accuracy': acc, 'log_loss': log_loss_}

train_data = load_data(train_data, tokenizer)
val_data = load_data(val_data, tokenizer)



class Gemma2ForSFT(LlamaForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False) # 最后一层
        self.post_init()

    # input>>(model)>>hidden_states>>(lm_head)>>logits
    def forward( # 重写forward方法
        self,
        input_ids= None,
        attention_mask= None,
        position_ids = None,
        past_key_values= None,
        inputs_embeds= None,
        labels= None,
        use_cache= None,
        output_attentions= None,
        output_hidden_states = None,
        return_dict= None,
        cache_position = None,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        hidden_states = outputs[0]
        # if self.config.pretraining_tp > 1:   // llama3才有，注释掉不然报错
        #     lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
        #     logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
        #     logits = torch.cat(logits, dim=-1)
        
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            # 对齐logits和labels
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            
            loss_fct = nn.CrossEntropyLoss()
            # loss_fct = focal_loss()
            
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)

            label_tokens_ids = torch.tensor(LABEL_IDS,device=shift_labels.device)
            index_mapping = {value.item(): idx for idx, value in enumerate(label_tokens_ids)}
            true_labels = shift_labels[torch.isin(shift_labels, label_tokens_ids)]
            true_labels = torch.tensor([index_mapping[label.item()] for label in true_labels], device=true_labels.device)
            true_logits = shift_logits[torch.isin(shift_labels, label_tokens_ids)][:,label_tokens_ids]
            loss = loss_fct(true_logits, true_labels)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=true_logits,
        )

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias='none',
    inference_mode=False,
    task_type=TaskType.CAUSAL_LM,
    # target_modules=['q_proj', 'k_proj', 'v_proj','o_proj'], 
    target_modules=['q_proj', 'k_proj', 'v_proj'], 
)

model = Gemma2ForSFT.from_pretrained(model_path, torch_dtype=torch.float16) # 加载模型

# torch.save(model.lm_head.weight, 'lm_head_weights.pth')

model.config.use_cache = False # 关闭 kvcache
model = prepare_model_for_kbit_training(model) # layer norm 层保留 FP32 精度，嵌入层以及 LM head 输出层保留 FP32 精度
model = get_peft_model(model, peft_config) # 在模型中添加 LoRA 层（参数使用 FP32）
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Layer: {name}, Trainable: {param.requires_grad}") # 打印每层信息
model.print_trainable_parameters()

args = TrainingArguments(
    output_dir="/root/output/Llama3_bnb_4bit",
    # all_data
    logging_steps=1000,
    save_steps=1000,

    overwrite_output_dir = True,
    evaluation_strategy = "steps", # epoch
    save_strategy = "steps",
    load_best_model_at_end=True,
    save_total_limit=5,
    logging_strategy="steps",

    warmup_steps=20,
    optim="adamw_8bit",
    learning_rate=2e-4,
    per_device_train_batch_size=1, # 2
    per_device_eval_batch_size=4, # 4
    gradient_accumulation_steps=1, # 2
    num_train_epochs=1,
    fp16=True,
    # load_best_model_at_end=True,
    greater_is_better = False,
    report_to="none",
)

trainer = Trainer(
    args=args,
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)

trainer.train(resume_from_checkpoint="/root/output/Llama3_bnb_4bit/checkpoint-1000")
# trainer.train()


import matplotlib.pyplot as plt  
train_history = trainer.state.log_history
train_losses = [(x['step'], x['loss']) for x in train_history if 'loss' in x]
val_losses = [(x['step'], x['eval_loss']) for x in train_history if 'eval_loss' in x]

train_steps, train_losses_values = zip(*train_losses)  
val_steps, val_losses_values = zip(*val_losses)  
# 绘制训练损失  
plt.plot(train_steps, train_losses_values, label='Training Loss', marker='o')  
# 绘制验证损失  
plt.plot(val_steps, val_losses_values, label='Validation Loss', marker='x')  
# 添加图例  
plt.legend()  
# 添加标题和坐标轴标签  
plt.title('Training and Validation Loss')  
plt.xlabel('Step')   
plt.ylabel('Loss')
plt.grid(True)  # 显示网格  
plt.savefig('./lossPNG/Llama3_bnb_4bit.png')
# 显示图表  
plt.show()

# 保存模型
trainer.model.save_pretrained("/root/result/Llama3_bnb_4bit")
tokenizer.save_pretrained("/root/result/Llama3_bnb_4bit")