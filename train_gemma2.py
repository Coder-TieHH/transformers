# gemma-2 is available from transformers>=4.42.3

import os
import copy
from dataclasses import dataclass

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    BitsAndBytesConfig,
    Gemma2ForSequenceClassification,
    GemmaTokenizerFast,
    Gemma2Config,
    PreTrainedTokenizerBase, 
    EvalPrediction,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
import pandas as pd
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from sklearn.metrics import log_loss, accuracy_score

@dataclass
class Config:
    output_dir: str = "/root/output/Gemma2_keep_train"
    checkpoint: str = "/root/result/gemma2/gemma2_newData_seq2cla"  # 4-bit quantized gemma-2-9b-instruct
    max_length: int = 1024
    # optim_type: str = "adamw_8bit"
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4  # global batch size is 8 
    per_device_eval_batch_size: int = 8
    n_epochs: int = 1
    # freeze_layers: int = 16  # there're 42 layers in total, we don't add adapters to the first 16 layers
    lr: float = 2e-4
    warmup_steps: int = 20
    lora_r: int = 16
    lora_alpha: float = lora_r * 2
    lora_dropout: float = 0.05
    lora_bias: str = "none"



config = Config()


training_args = TrainingArguments(
    output_dir="/root/output/Gemma2_keep_train",
    overwrite_output_dir=True,
    evaluation_strategy = "steps", # epoch
    save_strategy = "steps",
    save_steps=2000,
    load_best_model_at_end=True,
    save_total_limit=5,
    logging_strategy="steps",
    logging_steps=1000,
    warmup_steps=20,
    # optim="adamw_8bit",
    learning_rate=2e-4,
    per_device_train_batch_size=1, # 2
    per_device_eval_batch_size=8, # 4
    gradient_accumulation_steps=4, # 2
    num_train_epochs=1,
    fp16=True,
    # load_best_model_at_end=True,
    # greater_is_better = False,
    report_to="none",
    # deepspeed='/root/CONFIG/ds_config.json'
)

lora_config = LoraConfig(
    r=config.lora_r,
    lora_alpha=config.lora_alpha,
    # only target self-attention
    target_modules=["q_proj", "k_proj", "v_proj"],
    # layers_to_transform=[i for i in range(42) if i >= config.freeze_layers],
    lora_dropout=config.lora_dropout,
    bias=config.lora_bias,
    task_type=TaskType.SEQ_CLS,
)

tokenizer = GemmaTokenizerFast.from_pretrained(config.checkpoint)
tokenizer.add_eos_token = True  # We'll add <eos> at the end
tokenizer.padding_side = "right"

model = Gemma2ForSequenceClassification.from_pretrained(
    config.checkpoint,
    num_labels=3,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.config.use_cache = True
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

# train_data = pd.read_csv("/root/DATA/ds_train_new.csv")
# val_data = pd.read_csv("/root/DATA/ds_val_new.csv")



class CustomTokenizer:
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizerBase, 
        max_length: int
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __call__(self, batch: dict) -> dict:
        prompt = ["<prompt>: " + self.process_text(t) for t in batch["prompt"]]
        response_a = ["\n\n<response_a>: " + self.process_text(t) for t in batch["response_a"]]
        response_b = ["\n\n<response_b>: " + self.process_text(t) for t in batch["response_b"]]
        texts = [p + r_a + r_b for p, r_a, r_b in zip(prompt, response_a, response_b)]
        tokenized = self.tokenizer(texts, max_length=self.max_length, truncation=True)
        # labels=[]
        # for a_win, b_win in zip(batch["winner_model_a"], batch["winner_model_b"]):
        #     if a_win:
        #         label = 0
        #     elif b_win:
        #         label = 1
        #     else:
        #         label = 2
        #     labels.append(label)
        labels = batch['label']
        return {**tokenized, "labels": labels}
        
    @staticmethod
    def process_text(text: str) -> str:
        return " ".join(eval(text, {"null": ""}))

encode = CustomTokenizer(tokenizer, max_length=config.max_length)
train_data = Dataset.from_csv("/root/DATA/ds_train_new.csv")
val_data = Dataset.from_csv("/root/DATA/ds_val_new.csv")
train_data = train_data.map(encode, batched=True)
val_data = val_data.map(encode, batched=True)

def compute_metrics(eval_preds: EvalPrediction) -> dict:
    preds = eval_preds.predictions
    labels = eval_preds.label_ids
    probs = torch.from_numpy(preds).float().softmax(-1).numpy()
    loss = log_loss(y_true=labels, y_pred=probs)
    acc = accuracy_score(y_true=labels, y_pred=preds.argmax(-1))
    return {"acc": acc, "log_loss": loss}


trainer = Trainer(
    args=training_args, 
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_data,
    eval_dataset=val_data,
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
)
# trainer.train(resume_from_checkpoint="/root/result/gemma2/gemma2_newData_seq2cla")
trainer.train()


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

plt.savefig('/root/lossPNG/gemma2_newData_seq2cla_keep.png')
# 显示图表  
plt.show()

trainer.model.save_pretrained("/root/result/gemma2/gemma2_newData_seq2cla_keep")
tokenizer.save_pretrained("/root/result/gemma2/gemma2_newData_seq2cla_keep")




