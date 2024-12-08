import json
import gc
from torch.utils.data import Dataset
from accelerate import Accelerator, init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, AutoModelForCausalLM
from trl import AutoModelForCausalLMWithValueHead
from transformers.optimization import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup
import torch
from torch.optim import AdamW
import time
from torch.utils.data import DistributedSampler
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from tqdm import tqdm
import copy
from datasets import load_dataset
from trl.trainer.utils import pad
from torch.nn import CrossEntropyLoss
from peft import get_peft_config, get_peft_model, AdaptionPromptConfig, TaskType
import os
import json
from typing import Dict, Tuple
import numpy as np
import random
from collections import defaultdict
import pickle
import shutil
from file_io import *
import os
import re


class LogMessage:
    def __init__(self, log_file, disable=False):
        self.log_file = log_file
        self.disable = disable

    def log(self, *message):
        message = ' '.join([str(i) for i in message])
        if not self.disable:
            current_time = "[" + time.strftime("%H:%M:%S") + "]"
            with open(self.log_file, "a") as file:
                file.write(current_time + " " + message + '\n')
            print(current_time + " " + message)


def download_prm():
    # data = load_dataset('xinlai/Math-Step-DPO-10K')['train']
    os.system('wget https://github.com/openai/prm800k/raw/refs/heads/main/prm800k/math_splits/test.jsonl')
    # print(data)


def format_qa(text):
    step_1_index = text.find('Step 1')
    if step_1_index == -1:
        return text, ''
    else:
        return text[:step_1_index], text[step_1_index + len('Step 1'):]


def built_example(messages, tokenizer):
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    q_ids = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
    labels = [-100] * len(q_ids) + ids[len(q_ids):]
    input_ids = ids
    return input_ids, labels


class TrainPPOData(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        question = self.data[index]['question']

        # answer = self.data[index]['solution']
        answer = str(self.data[index]['answer'])
        messages = [{'role': 'user', 'content': question}]
        input_ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        labels = input_ids
        return torch.tensor(input_ids[:2048]), torch.tensor(labels[:2048]), answer

    def collate_fn(self, batch):
        input_ids, labels, answers = zip(*batch)
        input_ids = pad(input_ids, padding_side='left', padding_value=self.tokenizer.pad_token_id)
        labels = pad(labels, padding_side='left', padding_value=-100)
        features = {
            "input_ids": input_ids,
            "labels": labels,
            "answers": answers
        }
        return features


class TrainGroupData(TrainPPOData):
    def __init__(self, data, tokenizer, num=10):
        self.data = data
        self.tokenizer = tokenizer
        self.num = num

    def __getitem__(self, index):
        question = self.data[index]['question']
        answer = str(self.data[index]['answer'])
        msg = [{'role': 'user', 'content': question}]
        q_ids = self.tokenizer.apply_chat_template(msg, tokenize=True, add_generation_prompt=True)
        input_ids = []
        rollout = []
        answers = []
        for so in self.data[index]['solution'][:self.num]:
            so = 'Lets think step by step.\n\nStep 1: ' + so
            msg = [{'role': 'user', 'content': question}, {'role': 'assistant', 'content': so}]
            ids, _ = built_example(msg, self.tokenizer)
            rollout.append(torch.tensor(ids[len(q_ids):len(q_ids) + 2048]))
            input_ids.append(torch.tensor(q_ids[:2048]))
            answers.append(answer)
        return input_ids, rollout, answers

    def collate_fn(self, batch):
        input_ids, rollout, answers = zip(*batch)
        input_ids = sum(input_ids, [])
        rollout = sum(rollout, [])
        answers = sum(answers, [])
        input_ids = pad(input_ids, padding_side='left', padding_value=self.tokenizer.pad_token_id)
        rollout = pad(rollout, padding_side='right', padding_value=self.tokenizer.pad_token_id)
        rollout = torch.cat([input_ids, rollout], dim=1)
        features = {
            "input_ids": input_ids,
            'rollout': rollout,
            "answers": answers
        }
        return features


def train_ppo():
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    from ppo import PolicyAndValueWrapper, PPOTrainerWrapper, PPOConfig

    # logger = LogMessage("log/ppo.log", disable=not accelerator.is_main_process)
    logger = LogMessage("log/ppo.log")
    logger.log("")
    logger.log("#" * 20)
    logger.log("PPO")

    save_path = 'models/Llama-3.2-1B-Instruct-PPO'

    model_name = 'models/Llama-3.2-1B-Instruct-SFT'
    reward_name = 'models/Llama-3.2-1B-Instruct-ORM'

    rollout_num = 10
    batch_size = 1
    mini_batch_size = 10
    gradient_accumulation_steps = 1

    config = PPOConfig(
        num_ppo_epochs=1,
        num_mini_batches=rollout_num * batch_size // mini_batch_size,
        per_device_train_batch_size=mini_batch_size,
        response_length=512,
        local_batch_size=rollout_num * batch_size * gradient_accumulation_steps,
        local_rollout_forward_batch_size=128,
        local_mini_batch_size=mini_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        missing_eos_penalty=None,
        kl_coef=1e-3,
        lam=1.0,
        output_dir=save_path
    )

    accelerator = Accelerator(gradient_accumulation_steps=1)

    value_model = AutoModelForSequenceClassification.from_pretrained(
        reward_name, num_labels=1, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
    )
    value_model.config.use_cache = False
    value_model.gradient_checkpointing_enable()
    # reward_model = AutoModelForSequenceClassification.from_pretrained(
    #     reward_name, num_labels=1, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
    # )
    reward_model = None
    policy = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
    )
    policy.config.use_cache = False
    policy.generation_config.pad_token_id = 128004
    policy.gradient_checkpointing_enable()

    ref_policy = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
    )
    ref_policy.eval()

    model = PolicyAndValueWrapper(policy, value_model)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = 128004

    # data = load_dataset('GAIR/o1-journey')['train']
    # data = read_jsonl('data/train_math_gsm.jsonl')
    data = read_jsonl('data/Llama-1B-sample10-2-new-filter.jsonl')
    logger.log('Data', len(data))
    dataset = TrainGroupData(data, tokenizer, num=rollout_num)
    accelerator.print(tokenizer.decode(dataset[0][0][0]))
    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, shuffle=True,
                                              batch_size=batch_size, num_workers=4)
    optimizer = AdamW(model.parameters(), lr=5e-6)
    # optimizer = AdamW(model.parameters(), lr=1e-5)
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=10)
    model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)
    # ref_policy, reward_model = accelerator.prepare(ref_policy, reward_model)
    # reward_model.cuda()
    ref_policy.cuda()

    trainer = PPOTrainerWrapper(config, tokenizer, accelerator)

    if accelerator.is_main_process:
        os.system('sh kill_gpu.sh')

    for epoch in range(3):
        logger.log(f'Training {epoch}')
        accelerator.wait_for_everyone()
        model.train()
        tk0 = tqdm(data_loader, total=len(data_loader), disable=not accelerator.is_main_process)
        for batch in tk0:
            # print(batch['input_ids'].size())
            ppo_batch = trainer.rollout(batch, model, ref_policy, None, tokenizer, clean=False)
            # input()
            trainer.train_step(ppo_batch, model, optimizer, scheduler, clean=False)
            # metrics = trainer.log(ppo_batch, clean=False)
            # accelerator.print(metrics)
            # del ppo_batch
            # torch.cuda.empty_cache()
            # gc.collect()
            # loss_report.append(accelerator.gather(loss).mean().item())
            # tk0.set_postfix(
            #     loss=sum(loss_report[-100:]) / len(loss_report[-100:]),
            #     kl=sum(kl[-100:]) / len(kl[-100:]),
            #     # acc=sum(acc) / len(acc)
            # )
        accelerator.wait_for_everyone()
        logger.log()
    accelerator.wait_for_everyone()

    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        f'{save_path}',
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        # state_dict=accelerator.get_state_dict(model),
    )
    tokenizer.save_pretrained(f'{save_path}')


if __name__ == '__main__':
    train_ppo()

