import json
import random

from torch.utils.data import Dataset
from accelerate import Accelerator, init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, AutoModelForCausalLM
from transformers.optimization import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup
import torch
from torch.optim import AdamW
import time
from torch.utils.data import DistributedSampler
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from collections import OrderedDict
from tqdm import tqdm
import copy
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from peft import get_peft_config, get_peft_model, AdaptionPromptConfig, TaskType, LoraConfig
import os
import json
from typing import Dict, Tuple
import numpy as np
from collections import defaultdict
import pickle
import shutil
from file_io import *
from norm import em_answer
import os
import bm25s
import re

os.environ['http_proxy'] = 'http://agent.baidu.com:8891'
os.environ['https_proxy'] = 'http://agent.baidu.com:8891'


def remove_span(text):
    pattern = r"'''''[^']*'''''(?:\s*\([^()]*\))?"
    cleaned_text = re.sub(pattern, '', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text


class BM25():
    def __init__(self, corpus):
        self.corpus = corpus
        corpus_text = [x['text'] for x in corpus]
        corpus_tokens = bm25s.tokenize(corpus_text, stopwords="en")
        self.retriever = bm25s.BM25()
        self.retriever.index(corpus_tokens)

    def sample(self, k=10):
        index = [i for i in range(len(self.corpus))]
        np.random.shuffle(index)
        index = index[:k]
        return [self.corpus[i] for i in index]

    def search(self, query, k=10):
        query_tokens = bm25s.tokenize(query)
        results, scores = self.retriever.retrieve(query_tokens, corpus=self.corpus, k=k)
        out = []
        for i in range(results.shape[1]):
            doc, score = results[0, i], scores[0, i]
            out.append(doc)
        return out


def build_gold():
    data = read_jsonl('data/quest/train.jsonl')
    data.extend(read_jsonl('data/quest/train_aug.jsonl'))
    data.extend(read_jsonl('data/quest/test.jsonl'))
    retriever = BM25(read_jsonl('data/quest/tmp/documents.jsonl'))
    print(len(data))
    seen = set()
    new_data = []
    for item in tqdm(data):
        question = item['query']
        if question in seen:
            continue
        else:
            seen.add(question)
        docs = item['docs']
        answer = "Here are some " + question + '\n\n'
        label_set = [em_answer(x) for x in docs]

        retrieved = [x for x in retriever.search(question, k=100) if em_answer(x['title']) not in label_set]
        sampled = [x for x in retriever.sample(k=100) if em_answer(x['title']) not in label_set]

        # retrieved = []
        # sampled = []

        for i, doc in enumerate(docs):
            if item['metadata']['attributions'] is None:
                desp = ''
            elif item['metadata']['attributions'][doc][0] is None:
                desp = ''
            else:
                desp = list(item['metadata']['attributions'][doc][0].values())[0]
                desp = " - " + remove_span(desp).strip()
            answer += f'Option {i + 1}: {doc}{desp}\n\n'
        answer += 'Thats all for now! If you have any more questions, feel free to ask!'
        item['solution'] = answer
        item['retrieved'] = retrieved
        item['sampled'] = sampled
        new_data.append(item)

    print(len(new_data))
    write_jsonl(new_data, 'data/quest_train.jsonl')
    # new_val = []
    # for item in tqdm(read_jsonl('data/quest/val.jsonl')):
    #     docs = item['docs']
    #     best_match = 0
    #     for cand in new_data:
    #         match = [int(doc in cand['docs']) for doc in docs]
    #         match = sum(match)
    #         best_match = max(best_match, match)
    #     if best_match / len(docs) < 0.5:
    #         print(best_match, len(docs))
    #         new_val.append(item)
    # print(len(new_val))
    # write_jsonl(new_val, 'data/quest_val.jsonl')


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


def first_k(text, k):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    return ' '.join(sentences[:k])


class TrainGenData(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        question = self.data[index]['query']
        question = "Name some " + question
        item = self.data[index]
        docs = item['docs']
        answer = "Here are some " + self.data[index]['query'] + '\n\n'
        for i, doc in enumerate(docs):
            if item['metadata']['attributions'] is None:
                desp = ''
            elif item['metadata']['attributions'][doc][0] is None:
                desp = ''
            else:
                desp = list(item['metadata']['attributions'][doc][0].values())[0]
                desp = " - " + remove_span(desp).strip()
            # answer += f'Option {i + 1}: {doc}{desp}\n\n'
            answer += f'{i + 1}. **{doc}**\n\n'
        # answer += 'Thats all for now! If you have any more questions, feel free to ask!'
        messages = [{'role': 'user', 'content': question}, {'role': 'assistant', 'content': answer}]
        ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        q_ids = self.tokenizer.apply_chat_template([{'role': 'user', 'content': question}], tokenize=True,
                                                   add_generation_prompt=True)
        labels = [-100] * len(q_ids) + ids[len(q_ids):]

        return torch.tensor(ids[:2048]), torch.tensor(labels[:2048])

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


class TrainCorrectData(TrainGenData):
    def __getitem__(self, index):
        question = self.data[index]['query']
        question = "Name some " + question
        item = self.data[index]
        docs = item['docs']
        answer = "Here are some " + self.data[index]['query'] + '\n\n'
        for i, doc in enumerate(docs):
            if item['metadata']['attributions'] is None:
                desp = ''
            elif item['metadata']['attributions'][doc][0] is None:
                desp = ''
            else:
                desp = list(item['metadata']['attributions'][doc][0].values())[0]
                desp = remove_span(desp).strip()

            it = 0
            while random.random() < 0.5:
                it += 1
                if it > 1:
                    break
                neg = random.choice(item['retrieved'])
                answer += f"{i + 1}. **{neg['title']}**\n\nWait, lets think about this. I remind that:\n"
                answer += neg['title'] + ' ' + remove_span(first_k(neg['text'].strip(), 1))
                answer += f'\nRecall that the question is {question}. Hmm... Seems this is not a correct one to the question. Let skip this!\n\n'

            answer += f'{i + 1}. **{doc}**\n\n'
            if len(desp) > 0:
                answer += f'Wait, lets think about this. I remind that:\n{doc} {desp}\nRecall that the question is {question}. Then this is a correct one!\n\n'

        messages = [{'role': 'user', 'content': question}, {'role': 'assistant', 'content': answer}]
        ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        q_ids = self.tokenizer.apply_chat_template([{'role': 'user', 'content': question}], tokenize=True,
                                                   add_generation_prompt=True)
        labels = [-100] * len(q_ids) + ids[len(q_ids):]
        return torch.tensor(ids[:4096]), torch.tensor(labels[:4096])


def train_quest():
    accelerator = Accelerator(gradient_accumulation_steps=2)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    logger = LogMessage("log/quest.log", disable=not accelerator.is_main_process)
    logger.log("")
    logger.log("#" * 20)
    logger.log("PRM")

    batch_size = 2

    step_token_id = 128002  # <|reserved_special_token_0|>
    good_token_id = 7839  # Good
    bad_token_id = 11717  # Bad

    save_path = 'models/Llama-3.2-3B-Instruct-Quest-Correct'

    model_name = 'meta-llama/Llama-3.2-3B-Instruct'
    # model_name = 'models/Llama-3.2-1B-Instruct-SFT'
    # AutoModelForSequenceClassification
    # AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        # num_labels=1
    )
    model.config.use_cache = False
    model.config.pad_token_id = 128004

    # peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=16, lora_alpha=32,
    #                          lora_dropout=0.1)
    # model = get_peft_model(model, peft_config)
    # model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = 128004

    # data = load_dataset('peiyi9979/Math-Shepherd')['train']
    # data = load_dataset('xinlai/Math-Step-DPO-10K')['train']
    # data = [x for x in data][100:]
    # data = [x for x in data if 'Step 1' in x['input']]
    data = read_jsonl('data/quest_train.jsonl')
    # data = load_dataset('GAIR/o1-journey')['train']
    # data = read_jsonl('data/Llama-1B-PRM.jsonl')
    # data = [x for x in data if x['acc'] > 0]
    logger.log('Data', len(data))
    # TrainGenData, TrainCorrectData
    dataset = TrainCorrectData(data, tokenizer)
    accelerator.print(tokenizer.decode(dataset[0][0]))
    # accelerator.print(dataset[0][0])
    # accelerator.print(dataset[0][1])
    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, shuffle=True,
                                              batch_size=batch_size, num_workers=4)
    optimizer = AdamW(model.parameters(), lr=1e-5)
    model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)
    if accelerator.is_main_process:
        os.system('sh kill_gpu.sh')

    for epoch in range(3):
        logger.log(f'Training {epoch}')
        accelerator.wait_for_everyone()
        model.train()
        tk0 = tqdm(data_loader, total=len(data_loader), disable=not accelerator.is_main_process)
        loss_report = []
        acc = []
        for batch in tk0:
            with accelerator.accumulate(model):
                # good_logits = model(input_ids=batch['good']).logits
                # bad_logits = model(input_ids=batch['bad']).logits
                # loss = F.binary_cross_entropy_with_logits(good_logits, torch.ones_like(good_logits))
                # loss += F.binary_cross_entropy_with_logits(bad_logits, torch.zeros_like(bad_logits))
                # acc.extend((good_logits.view(-1) > 0).float().cpu().tolist())
                # acc.extend((bad_logits.view(-1) < 0).float().cpu().tolist())

                out = model(input_ids=batch['input_ids'], labels=batch['labels'])
                # logits = out.logits
                loss = out.loss

                # i_out = model(input_ids=batch['i_input_ids'], labels=batch['i_labels'])
                # i_logits = i_out.logits
                # policy_chosen_logps = torch.gather(logits.log_softmax(-1), dim=2, index=batch['labels'].unsqueeze(2)).squeeze(2)
                # policy_rejected_logps = torch.gather(i_logits.log_softmax(-1), dim=2, index=batch['i_labels'].unsqueeze(2)).squeeze(2).sum(-1)
                # po_loss, _, _ = simpo_loss(policy_chosen_logps, policy_rejected_logps)
                # loss = loss + po_loss

                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                optimizer.zero_grad()
            # logits[:, :, [good_token_id, bad_token_id]]
            loss_report.append(accelerator.gather(loss).mean().item())
            tk0.set_postfix(
                loss=sum(loss_report[-100:]) / len(loss_report[-100:]),
                # acc=sum(acc) / len(acc)
            )
        # if accelerator.is_main_process:
        #     eval_results = eval(accelerator.unwrap_model(model), tokenizer, dev, docs, tree, base=base)
        #     logger.log(eval_results)
        accelerator.wait_for_everyone()
        logger.log()
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        # unwrapped_model = unwrapped_model.merge_and_unload()
        unwrapped_model.save_pretrained(
            f'{save_path}',
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )
        tokenizer.save_pretrained(f'{save_path}')

    accelerator.wait_for_everyone()


def remove_tag(text):
    return re.sub(r'^\d+\.\s+', '', text)


def get_major(outputs):
    split_outputs = [[em_answer(remove_tag(x)) for x in out.split("\n\n") if len(x) > 0] for out in outputs]
    vote = defaultdict(int)
    for out in split_outputs:
        out = set(out)
        for x in out:
            vote[x] += 1
    vote_results = [sum([vote[x] for x in out]) / len(out) for out in split_outputs]
    pick = np.argmax(vote_results)
    return outputs[pick]


def filter_answer(output):
    num2line = OrderedDict()
    for line in output.split('\n'):
        if len(line.split()) == 0:
            continue
        tag: str = line.split()[0].replace('.', '')
        if tag.isdigit():
            num2line[int(tag)] = line
    return "\n\n".join(list(num2line.values()))


def predict(data, seed=0, model_name='meta-llama/Llama-3.2-1B-Instruct'):
    from norm import em_answer
    gpu_id = seed % 8
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
    )
    model = model.to(f'cuda:{gpu_id}')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bar = tqdm(data)
    metric = defaultdict(list)
    if seed == 0:
        os.system('sh kill_gpu.sh')
    for item in bar:
        problem = item['query']
        # label = item['answer']
        prompt = problem

        messages = [{"role": "user", "content": "Name some " + prompt},
                    {'role': 'assistant', 'content': f"Here are some {prompt}\n\n1."}]
        input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        input_ids = input_ids[:-1]

        out = model.generate(
            input_ids=torch.tensor([input_ids]).to(f'cuda:{gpu_id}'),
            max_new_tokens=4096,
            do_sample=True,
            temperature=1,
            top_p=0.9,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
        )

        out = [x[len(input_ids):] for x in out]
        # outputs = tokenizer.decode(out)
        outputs = [tokenizer.decode(x, skip_special_tokens=True) for x in out]

        print(outputs[0])

        # outputs = outputs[0]["generated_text"][-1]['content']
        outputs = [filter_answer('1. ' + x) for x in outputs]
        outputs = [get_major(outputs)]
        output = [em_answer(a) for a in outputs[0].split('\n') if len(a.strip()) != 0]
        label = [em_answer(a) for a in item['docs']]
        cat_output = " ".join(output)
        correctness = [int(a in cat_output) for a in label]
        
        # print("Positive", sum(correctness))
        # print("Precision", sum(correctness) / len(output))
        # print("Recall", sum(correctness) / len(label))
        # input()
        metric['Positive'].append(sum(correctness))
        metric['Precision'].append(sum(correctness) / max(len(output), len(correctness)))
        metric['Recall'].append(sum(correctness) / len(label))

        bar.set_postfix(acc=sum(metric['Precision']) / len(metric['Precision']))
    return [[k, v] for k, v in metric.items()]


def test_quest():
    data = read_jsonl('data/quest_val.jsonl')
    model_name = 'models/Llama-3.2-1B-Instruct-Quest-Correct'
    # model_name = 'meta-llama/Llama-3.2-1B-Instruct'
    out = mp(predict, data, processes=16, model_name=model_name)
    metric = defaultdict(list)

    for k, v in out:
        metric[k].extend(v)
    for k, v in metric.items():
        print(k, sum(v) / len(v))


if __name__ == '__main__':
    # build_gold()
    train_quest()
    # test_quest()
