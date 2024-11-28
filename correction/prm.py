import json
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


class TrainRMData(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        question = self.data[index]['question']
        prompt = "\n\n".join(self.data[index]['prompt'])
        label = "\n\n".join(self.data[index]['label'])

        prompt = [{'role': 'user', 'content': question}, {'role': 'assistant', 'content': prompt}]
        label = [{'role': 'user', 'content': question}, {'role': 'assistant', 'content': label}]

        prompt_ids = self.tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=False)
        label_ids = self.tokenizer.apply_chat_template(label, tokenize=True, add_generation_prompt=False)

        if len(prompt_ids) != len(label_ids):
            print(self.data[index])
        assert len(prompt_ids) == len(label_ids)
        label_ids = [-100 if i != 128002 else j for i, j in zip(prompt_ids, label_ids)]
        return torch.tensor(prompt_ids[:2048]), torch.tensor(label_ids[:2048])

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        input_ids, labels = zip(*batch)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        features = {
            "input_ids": input_ids,
            "labels": labels
        }
        return features


class TrainORMData(TrainRMData):
    def __getitem__(self, index):
        question = self.data[index]['question']
        good_solution = self.data[index]['good_solution']
        bad_solution = self.data[index]['bad_solution']

        input_ids = []
        for so in good_solution + bad_solution:
            so = 'Lets think step by step.\n\nStep 1: ' + so
            msg = [{'role': 'user', 'content': question}, {'role': 'assistant', 'content': so}]
            ids, _ = built_example(msg, self.tokenizer)
            input_ids.append(torch.tensor(ids[:1024]))
        labels = [1] * len(good_solution) + [0] * len(bad_solution)
        return input_ids, labels

    def collate_fn(self, batch):
        input_ids, labels = zip(*batch)
        input_ids = sum(input_ids, [])
        labels = sum(labels, [])
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.tensor(labels)
        features = {
            "input_ids": input_ids,
            'labels': labels,
        }
        return features


def extract_pair(logits, labels):
    loss_mask = labels != -100
    labels[labels == -100] = 0
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    sample_logps = (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    sample_logps = sample_logps.view(2, -1)
    sample_logps = sample_logps.view(-1, 2).transpose(0, 1)
    policy_chosen_logps = sample_logps[0]
    policy_rejected_logps = sample_logps[1]
    return policy_chosen_logps, policy_rejected_logps


def simpo_loss(
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        beta=10, gamma_beta_ratio=0.3, label_smoothing=0, loss_type="sigmoid"
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    logits = pi_logratios - gamma_beta_ratio

    if loss_type == "sigmoid":
        losses = (
                -F.logsigmoid(beta * logits) * (1 - label_smoothing)
                - F.logsigmoid(-beta * logits) * label_smoothing
        )
        print(losses)
    elif loss_type == "hinge":
        losses = torch.relu(1 - beta * logits)
    else:
        raise ValueError(
            f"Unknown loss type: {loss_type}. Should be one of ['sigmoid', 'hinge']"
        )

    chosen_rewards = beta * policy_chosen_logps.detach()
    rejected_rewards = beta * policy_rejected_logps.detach()
    loss = losses.mean()
    return loss, chosen_rewards, rejected_rewards


def split_steps(text):
    steps = re.split(r"Step \d+:", text)
    steps = [step.strip() for step in steps if step.strip()]
    think = 'Lets think step by step.'
    conclusion = 'In conclusion, the final answer is:'
    if steps[-1].count(conclusion) == 1:
        last, answer = steps[-1].split(conclusion)
        steps = steps[:-1] + [last.strip(), conclusion + answer]
    solution = []
    for i, step in enumerate(steps):
        if think in step or conclusion in step:
            solution.append(step)
        else:
            solution.append(f"Step {i}: " + step)
    return solution


def remove_label(texts):
    new = []
    for t in texts:
        t = t.strip()
        if t.endswith('+') or t.endswith('-'):
            t = t[:-1]
        new.append(t)
    return new


class TrainGenData(TrainRMData):
    def __getitem__(self, index):
        question = self.data[index]['question']

        # solution = self.data[index]['solution']
        # solution = random.choice(solution)
        # solution = 'Lets think step by step.\n\nStep 1:' + solution.replace('<|eot_id|>', '')
        # answer = '\n\n'.join(split_steps(solution))

        solution = self.data[index]['longCOT']
        answer = solution

        messages = [{'role': 'user', 'content': question}, {'role': 'assistant', 'content': answer}]
        ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        q_ids = self.tokenizer.apply_chat_template([{'role': 'user', 'content': question}], tokenize=True,
                                                   add_generation_prompt=True)
        labels = [-100] * len(q_ids) + ids[len(q_ids):]

        return torch.tensor(ids[:2048]), torch.tensor(labels[:2048])


class TrainCorrectData(TrainRMData):
    def __getitem__(self, index):
        question = self.data[index]['question']

        # cut = random.choice([k for k in self.data[index]['revised']])
        # prefix = self.data[index]['solution'][:int(cut)]
        # correct = [self.data[index]['solution'][int(cut):]]
        # incorrect = [[self.data[index]['revised'][cut].strip()]]
        # prefix_value = 0.5
        # base_value = 0

        prefix = self.data[index]['prefix']
        correct = self.data[index]['correct']
        incorrect = self.data[index]['incorrect']
        prefix_value = self.data[index]['prefix_value']
        base_value = self.data[index]['base_vale']

        nli = self.data[index]['nli']

        contradict_correct = []
        neutral_correct = []
        for cr in nli:
            for icr in nli[cr]:
                if nli[cr][icr] == 'Contradiction':
                    contradict_correct.append(cr)
                if nli[cr][icr] == 'Neutral':
                    neutral_correct.append(cr)

        if len(prefix) > 0:
            prefix[0] = prefix[0].replace('Step 1:', 'Step 1: ')

        max_att = 1
        max_dep = 100

        seed = random.random()
        seed = seed
        # seed = -10
        prefix_value = prefix_value - 0.2

        if len(neutral_correct) == 0 and len(contradict_correct) == 0:
            seed = -100
        # else:
        #     seed = 100

        seed = -100

        if seed < prefix_value:
            correct: list = random.choice(correct)
            answer = prefix + correct
        else:
            if len(contradict_correct) > 0:
                correct = [x for x in correct if x[0] in contradict_correct]
            correct: list = random.choice(correct)
            contradict_incorrect = [x for x in incorrect if nli[correct[0]][x[0]] == 'Contradiction']
            if len(contradict_incorrect) == 0:
                contradict_incorrect = incorrect
            reject = random.choice(contradict_incorrect)
            max_dep = min(max_dep, len(reject))
            cut_at = random.choice([i for i in range(1, max_dep + 1)])
            reject = reject[:cut_at]
            if correct[0].startswith('Step') and ':' in correct[0][:10]:
                back_step = correct[0].split(':')[0]
            else:
                back_step = 'previous step'
            if prefix[-1].startswith('Step') and ':' in prefix[-1][:10]:
                back_to = ' to ' + prefix[-1].split(':')[0]
            else:
                back_to = ''
            # if nli[correct[0]][reject[0]] == 'Contradiction':
            #     answer = prefix + reject + [
            #         f"Wait, this seems to be incorrect. Lets go back{back_to} and correct {back_step}."] + correct
            # else:
            #     answer = prefix + reject + [
            #         f"Wait, this seems to be incorrect. Lets go back{back_to} and re-try {back_step}."] + correct

            answer = prefix + reject + [f"Wait, this seems to be incorrect. Lets correct it."] + correct

        answer = '\n\n'.join(answer)
        answer = "Let's solve this problem step by step, and I will reflect on each step and promptly backtrack and correct.\n\n" + answer

        messages = [{'role': 'user', 'content': question}, {'role': 'assistant', 'content': answer}]
        ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        q_ids = self.tokenizer.apply_chat_template([{'role': 'user', 'content': question}], tokenize=True,
                                                   add_generation_prompt=True)
        labels = [-100] * len(q_ids) + ids[len(q_ids):]

        if not seed < prefix_value:
            # reject = prefix + reject + [f"Wait, this seems to be incorrect. Lets correct it."]
            reject = reject
            reject_ids = self.tokenizer.encode("\n\n".join(reject))[5:-2]
            for i in range(len(labels)):
                if labels[i:i + len(reject_ids)] == reject_ids:
                    labels = labels[:i] + [-100] * len(reject_ids) + labels[i + len(reject_ids):]
                    break
            else:
                print('Not found reject part')

        return torch.tensor(ids[:8000]), torch.tensor(labels[:8000])


class TrainPairData(TrainRMData):
    def __getitem__(self, index):
        question = self.data[index]['prompt']
        answer = self.data[index]['initial_reason_steps'] + self.data[index]['full_chosen']
        messages = [{'role': 'user', 'content': question}, {'role': 'assistant', 'content': answer}]
        ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)

        i_answer = self.data[index]['initial_reason_steps'] + self.data[index]['full_rejected']
        i_messages = [{'role': 'user', 'content': question}, {'role': 'assistant', 'content': i_answer}]
        i_ids = self.tokenizer.apply_chat_template(i_messages, tokenize=True, add_generation_prompt=False)

        return torch.tensor(ids[:2048]), torch.tensor(ids[:2048]), torch.tensor(i_ids[:2048]), torch.tensor(
            i_ids[:2048])

    def collate_fn(self, batch):
        input_ids, labels, i_input_ids, i_labels = zip(*batch)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        i_input_ids = pad_sequence(i_input_ids, batch_first=True, padding_value=0)
        i_labels = pad_sequence(i_labels, batch_first=True, padding_value=-100)
        features = {
            "input_ids": input_ids,
            "labels": labels,
            "i_input_ids": i_input_ids,
            "i_labels": i_labels,
        }
        return features


def train_prm():
    accelerator = Accelerator(gradient_accumulation_steps=8)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    logger = LogMessage("log/prm.log", disable=not accelerator.is_main_process)
    logger.log("")
    logger.log("#" * 20)
    logger.log("PRM")

    batch_size = 1

    step_token_id = 128002  # <|reserved_special_token_0|>
    good_token_id = 7839  # Good
    bad_token_id = 11717  # Bad

    good_token_id = 15571  # Good (First token)
    bad_token_id = 17519  # Bad

    save_path = 'models/Llama-3.2-1B-Instruct-ORM'

    # model_name = 'meta-llama/Llama-3.2-1B-Instruct'
    # save_path = 'models/Llama-3.2-1B-Instruct-Value-Rollout'
    # model_name = 'meta-llama/Llama-3.2-1B'
    model_name = 'models/Llama-3.2-1B-Instruct'
    # AutoModelForSequenceClassification
    # AutoModelForCausalLM, AutoModelForCausalLMWithValueHead
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        num_labels=1
    )
    model.config.use_cache = False
    model.config.pad_token_id = 128004
    # model.gradient_checkpointing_enable()

    # old_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2")
    # old_model = accelerator.prepare(old_model)
    # old_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = 128004

    from value_data import TrainGAEData, gae_loss, TrainRollOutData, TrainRefRollOutData

    # data = load_dataset('peiyi9979/Math-Shepherd')['train']
    # data = load_dataset('xinlai/Math-Step-DPO-10K')['train']
    # data = [x for x in data][100:]
    # data = [x for x in data if 'Step 1' in x['input']]
    data = read_jsonl('data/Llama-1B-ORM.jsonl')
    # data = read_jsonl('data/Llama-1B-Branch-NLI.jsonl')
    # data = read_jsonl('data/Llama-1B-sample10-Gold.jsonl')
    # data = read_jsonl('data/Llama-1B-Value-Neg.jsonl')
    # data = load_dataset('GAIR/o1-journey')['train']
    # data = read_jsonl('data/Llama-3B-greedy.jsonl')
    # data = [x for x in data if x['acc'] > 0]
    logger.log('Data', len(data))
    # dataset = TrainCorrectData(data, tokenizer)
    # dataset = TrainGenData(data, tokenizer)
    # dataset = TrainRefRollOutData(data, tokenizer, num=8)
    dataset = TrainORMData(data, tokenizer)
    accelerator.print(tokenizer.decode(dataset[0][0][0]))
    # accelerator.print(dataset[0][1])
    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, shuffle=True,
                                              batch_size=batch_size, num_workers=4)
    optimizer = AdamW(model.parameters(), lr=5e-6)
    # optimizer = AdamW(model.parameters(), lr=1e-5)
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=10)
    model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)
    if accelerator.is_main_process:
        os.system('sh kill_gpu.sh')

    for epoch in range(4):
        logger.log(f'Training {epoch}')
        accelerator.wait_for_everyone()
        model.train()
        tk0 = tqdm(data_loader, total=len(data_loader), disable=not accelerator.is_main_process)
        loss_report = []
        kl = [0]
        for batch in tk0:
            with accelerator.accumulate(model):
                logits = model(batch['input_ids']).logits
                labels = batch['labels'].unsqueeze(1)
                log_diff = -F.logsigmoid(logits - logits.t())
                true_diff = (labels - labels.t()) > 0.1
                log_diff[~true_diff] = 0.
                loss = log_diff.sum() / true_diff.sum()

                loss += 0.05 * torch.mean((logits.sum()) ** 2)

                # r_good = model(batch['good_ids'], attention_mask=batch['good_ids'].ne(128004)).logits
                # r_bad = model(batch['bad_ids'], attention_mask=batch['bad_ids'].ne(128004)).logits
                # loss = -F.logsigmoid(r_good - r_bad).mean()
                # loss += 0.1 * torch.mean((r_good + r_bad) ** 2)

                # out = model(input_ids=batch['input_ids'], labels=batch['labels'])
                # loss = out.loss

                # out = model(input_ids=batch['input_ids'])
                # loss = gae_loss(out, batch['index'], batch['labels'])

                # with torch.no_grad():
                #     old_out = old_model(input_ids=batch['input_ids'], labels=batch['labels'])
                #     old_probs = old_out.logits.log_softmax(dim=-1).detach()
                #     probs = out.logits.log_softmax(dim=-1)
                #     kl_loss = F.kl_div(probs, old_probs, log_target=True, reduction='none')
                #     kl_loss = kl_loss.sum(dim=-1)
                #     kl_loss = kl_loss * batch['labels'].ne(-100).float()
                #     kl_loss = kl_loss.sum() / batch['labels'].ne(-100).sum()
                #     kl.append(accelerator.gather(kl_loss).mean().item())
                # loss = loss + 0.1 * kl_loss

                # i_out = model(input_ids=batch['i_input_ids'], labels=batch['i_labels'])
                # i_logits = i_out.logits
                # policy_chosen_logps = torch.gather(logits.log_softmax(-1), dim=2, index=batch['labels'].unsqueeze(2)).squeeze(2)
                # policy_rejected_logps = torch.gather(i_logits.log_softmax(-1), dim=2, index=batch['i_labels'].unsqueeze(2)).squeeze(2).sum(-1)
                # po_loss, _, _ = simpo_loss(policy_chosen_logps, policy_rejected_logps)
                # loss = loss + po_loss

                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
            # logits[:, :, [good_token_id, bad_token_id]]
            loss_report.append(accelerator.gather(loss).mean().item())
            tk0.set_postfix(
                loss=sum(loss_report[-100:]) / len(loss_report[-100:]),
                kl=sum(kl[-100:]) / len(kl[-100:]),
                # acc=sum(acc) / len(acc)
            )
        # if accelerator.is_main_process:
        #     eval_results = eval(accelerator.unwrap_model(model), tokenizer, dev, docs, tree, base=base)
        #     logger.log(eval_results)
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
    train_prm()
