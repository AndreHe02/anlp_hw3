import json
from torch.utils.data import Dataset
from accelerate import Accelerator, init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, AutoModelForCausalLM
from transformers.optimization import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup
import torch
from torch.optim import AdamW
import time
from torch.utils.data import DistributedSampler
from torch.nn.utils.rnn import pad_sequence
from norm import math_normalizer as math_norm
from trl import AutoModelForCausalLMWithValueHead
import torch.nn.functional as F
from tqdm import tqdm
import copy
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from peft import get_peft_config, get_peft_model, AdaptionPromptConfig, TaskType
import os
import json
from typing import Dict, Tuple
import numpy as np
import random
from collections import defaultdict
from norm import get_majority_vote, numbered
import pickle
import shutil
from file_io import *
import os
import re


def split_steps(text):
    think = 'Lets think step by step.'
    conclusion = 'In conclusion, the final answer is'
    steps = re.findall(r'(Step \d+:.*?)(?=Step \d+:|$)', text, re.DOTALL)
    if not steps:
        steps = [text.strip()]
    else:
        steps = [step.strip() for step in steps if step.strip()]

    if steps and conclusion in steps[-1]:
        last_step = steps[-1]
        parts = last_step.split(conclusion, 1)
        if len(parts) >= 2:
            parts[1] = parts[1].split('\n')[0]
            steps = steps[:-1] + [parts[0].strip(), conclusion + parts[1]]

    solution = []
    for step in steps:
        solution.append(step.replace('<|eot_id|>', ''))

    return solution


class TrainGAEData(Dataset):
    def __init__(self, data, tokenizer, num=10):
        self.data = data
        self.tokenizer = tokenizer
        self.num = num

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        question = self.data[index]['question']
        all_solution = self.data[index]['all_solution']
        answer = self.data[index]['answer']
        reward = [int(math_norm(s) == str(answer)) for s in all_solution]
        batch_input_ids = []
        batch_idx = []
        batch_labels = []
        random.shuffle(all_solution)
        for n, solution in enumerate(all_solution[:self.num]):
            solution = 'Lets think step by step.\n\nStep 1:' + solution
            solution = split_steps(solution)
            solution = ["Let's solve this problem step by step."] + solution
            input_text = ""
            output_text = ""
            labels = []
            for step in solution:
                input_text += step + '\n\n' + "Verification: This step is<|reserved_special_token_0|>\n\n"
                output_text += step + '\n\n' + "Verification:This step is<|reserved_special_token_0|>\n\n"
                labels.append(reward[n])
            messages = [{'role': 'user', 'content': question}, {'role': 'assistant', 'content': input_text}]
            input_ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
            idx = [i for i in range(len(input_ids)) if input_ids[i] == 128002]
            batch_input_ids.append(torch.tensor(input_ids[:2048]))
            batch_idx.append(idx)
            batch_labels.append(labels)

        return batch_input_ids, batch_labels, batch_idx

    def collate_fn(self, batch):
        input_ids, labels, idx = zip(*batch)
        input_ids = sum(input_ids, [])
        labels = sum(labels, [])
        index = sum(idx, [])
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        features = {
            "input_ids": input_ids,
            'labels': labels,
            'index': index,
        }
        return features


def gae_loss(lm_out, index, labels):
    lm_logits, lm_loss, value = lm_out
    bs = value.size(0)
    pred_list = []
    label_list = []
    for i in range(bs):
        pred_list.append(value[i, index[i]])
        label_list.extend(labels[i])
    pred_list = torch.cat(pred_list, dim=0)
    label_list = torch.tensor(label_list, device=value.device)
    loss = (pred_list - label_list) ** 2
    return torch.mean(loss)


class TrainRollOutData(Dataset):
    def __init__(self, data, tokenizer, num=10):
        self.data = data
        self.tokenizer = tokenizer
        self.num = num

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        question = self.data[index]['question']
        solution = self.data[index]['solution']
        correctness = self.data[index]['correctness']

        answer = self.data[index]['answer']
        batch_input_ids = []
        batch_idx = []
        batch_labels = []
        input_text = ""
        output_text = ""
        labels = []
        for n, step in enumerate(solution):
            input_text += step + '\n\n' + "Verification: This step is<|reserved_special_token_0|>\n\n"
            output_text += step + '\n\n' + "Verification:This step is<|reserved_special_token_0|>\n\n"
            labels.append(correctness[n])
        messages = [{'role': 'user', 'content': question}, {'role': 'assistant', 'content': input_text}]
        input_ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        idx = [i for i in range(len(input_ids)) if input_ids[i] == 128002]
        batch_input_ids.append(torch.tensor(input_ids[:2048]))
        batch_idx.append(idx)
        batch_labels.append(labels)
        return batch_input_ids, batch_labels, batch_idx

    def collate_fn(self, batch):
        input_ids, labels, idx = zip(*batch)
        input_ids = sum(input_ids, [])
        labels = sum(labels, [])
        index = sum(idx, [])
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        features = {
            "input_ids": input_ids,
            'labels': labels,
            'index': index,
        }
        return features


class TrainRefRollOutData(TrainRollOutData):
    def __getitem__(self, index):
        question = self.data[index]['question']
        solution = self.data[index]['solution']
        correctness = self.data[index]['correctness']
        answer = self.data[index]['answer']
        # answer = numbered(answer)

        positive = [x for x in self.data[index]['all_solution'] if math_norm(x) == str(answer)]
        positive = random.choice(positive)
        positive = 'Step 1:' + positive
        batch_input_ids = []
        batch_idx = []
        batch_labels = []
        input_text = ""
        output_text = ""
        labels = []
        for n, step in enumerate(solution):
            input_text += step + '\n\n' + "Verification: This step is<|reserved_special_token_0|>\n\n"
            output_text += step + '\n\n' + "Verification:This step is<|reserved_special_token_0|>\n\n"
            labels.append(correctness[n])
        solution = '\n\n'.join(solution)
        prompt = (
            "Given a question and a correct reference solution, judge the correctness of each step of another solution."
            f"\n\nQuestion: {question}\n\nReference Correct Solution: {positive}\n\n"
            f"Recall the question is: \n\n{question}.\n\n"
            f"Based on the reference solution, compare each step and judge the correctness of the following solution:\n\n{solution}."
            f"You maybe compare it with reference step by step. Lets do it!")
        messages = [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': input_text}]
        input_ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        idx = [i for i in range(len(input_ids)) if input_ids[i] == 128002]
        labels = labels[:len(idx)]
        batch_input_ids.append(torch.tensor(input_ids[:8000]))
        batch_idx.append(idx)
        batch_labels.append(labels)
        return batch_input_ids, batch_labels, batch_idx


def eval():
    model_name = 'models/Llama-3.2-1B-Instruct-Value-Ref'
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        # num_labels=1
    )
    model = model.cuda()
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data = read_jsonl('data/gsm8k-1b-Value.jsonl')
    data = [x for x in data if str(numbered(x['answer'])) != math_norm(x['solution'][-1])]
    dataset = TrainRefRollOutData(data, tokenizer)

    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, shuffle=False,
                                              batch_size=32, num_workers=4)

    tk0 = tqdm(data_loader, total=len(data_loader))
    all_mse = []
    all_adv = []
    os.system('sh kill_gpu.sh')
    for batch in tk0:
        with torch.no_grad():
            out = model(input_ids=batch['input_ids'].cuda())
            lm_logits, lm_loss, value = out
            value = value.cpu()
            bs = value.size(0)
            for i in range(bs):
                lbs = torch.tensor(batch['labels'][i])
                pred = value[i, batch['index'][i]]
                mse = (pred - lbs) ** 2
                all_mse.append(mse.mean().item())

                lbs_diff = lbs[1:] - lbs[:-1]
                pred_diff = pred[1:] - pred[:-1]
                mse = (pred_diff - lbs_diff) ** 2
                all_adv.append(mse.mean().item())

    print(sum(all_mse) / len(all_mse))
    print(sum(all_adv) / len(all_adv))


if __name__ == '__main__':
    eval()
