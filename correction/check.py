from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from datasets import load_dataset
import torch
from file_io import *
from tqdm import tqdm
from collections import Counter
from norm import math_normalizer as math_norm
from norm import get_majority_vote, numbered
from tqdm import tqdm
import numpy as np
import random

import re
import os

os.environ['http_proxy'] = 'http://agent.baidu.com:8891'
os.environ['https_proxy'] = 'http://agent.baidu.com:8891'

def show_nli():
    data = read_jsonl('data/Llama-3B-Branch-NLI.jsonl')
    for item in data:
        prefix = item['prefix']
        correct = item['correct']
        incorrect = item['incorrect']
        prefix_value = item['prefix_value']
        base_value = item['base_vale']
        nli = item['nli']

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
        max_dep = 3

        seed = random.random()
        seed = seed
        # seed = -10
        prefix_value = prefix_value - 0.2
        if len(neutral_correct) == 0 and contradict_correct == 0:
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
            # max_dep = min(max_dep, len(reject))
            # cut_at = random.choice([i for i in range(1, max_dep + 1)])
            # reject = reject[:cut_at]
            if correct[0].startswith('Step') and ':' in correct[0][:10]:
                back_step = correct[0].split(':')[0]
            else:
                back_step = 'previous step'
            if prefix[-1].startswith('Step') and ':' in prefix[-1][:10]:
                back_to = ' to ' + prefix[-1].split(':')[0]
            else:
                back_to = ''
            if nli[correct[0]][reject[0]] == 'Contradiction':
                answer = prefix + reject + [f"Wait, lets recheck the answer... Hmm... Seems to be some problem... Lets go back{back_to} and re-try {back_step}."] + correct
            else:
                answer = prefix + reject + [f"Wait, lets recheck the answer... Hmm... Maybe we can do in a different way... Lets go back{back_to} and re-try {back_step}."] + correct
        
        answer = '\n\n'.join(answer)
        answer = "Let's solve this problem step by step, and I will reflect on each step and promptly backtrack and correct.\n\n" + answer
        print(len(contradict_correct))
        input()

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


class RollOut:
    def __init__(self, model, tokenizer, num=1, name='llama'):
        self.model = model
        self.tokenizer = tokenizer
        self.num = num
        self.name = name.lower()

    def rollout(self, input_ids, num=None, max_new_tokens=256):
        num = num or self.num
        out = self.model.generate(
            input_ids=torch.tensor([input_ids]).to(self.model.device),
            max_new_tokens=max_new_tokens,
            do_sample=num != 1,
            temperature=1.0,
            top_p=0.9,
            num_return_sequences=num,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        out = [x[len(input_ids):] for x in out]
        outputs = [self.tokenizer.decode(x, skip_special_tokens=True) for x in out]
        return outputs

    def generate(self, question, prefix=None, add_nn=False, max_new_tokens=256):
        if prefix is None:
            messages = [{"role": "user", "content": question}]
            input_ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        else:
            messages = [{"role": "user", "content": question},
                        {"role": "assistant", "content": prefix}]
            input_ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
            if 'qwen' in self.name:
                input_ids = input_ids[:-2]
            else:
                input_ids = input_ids[:-1]
            if add_nn and 'qwen' not in self.name:
                input_ids = input_ids + [271]
        return self.rollout(input_ids, 1, max_new_tokens)[0]


def pred(data, seed):
    model_name = 'meta-llama/Llama-3.2-3B-Instruct'
    gpu_id = seed % 8
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    )
    model = model.to(f'cuda:{gpu_id}')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # data = read_jsonl('data/Llama-1B-greedy.jsonl')
    rollout = RollOut(model, tokenizer, num=1)
    output = []
    for item in tqdm(data):
        if item['acc'] == 0:
            continue
        question = item['question']
        solution = 'Lets think step by step.\n\nStep 1:' + item['solution']
        solution = split_steps(solution)
        revised_step = {}
        for i in range(1, len(solution) - 1):
            step = solution[i]
            prompt = ("Please modify the following math reasoning step to make it to be **incorrect**. You can consider add some calculation error. Just directly modify the content to make this step become incorrect, DO NOT add new sentence. Make the mistake looks reasonable.\n\n"
                      f"{step}")
            prefix = f"Original step: {step}\n\nRevised step:"
            out = rollout.generate(prompt, prefix)
            revised_step[i] = out
            # print(step)
            # print(out)
            # input()
        item['solution'] = solution
        item['revised'] = revised_step
        output.append(item)
    return output


def insert():
    model_name = 'meta-llama/Llama-3.2-3B-Instruct'
    # model_name = 'meta-llama/Llama-3.1-8B-Instruct'
    gpu_id = seed % 8
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        token="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"
    )
    model = model.to(f'cuda:{gpu_id}')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # data = read_jsonl('data/Llama-1B-greedy.jsonl')
    rollout = RollOut(model, tokenizer, num=1)
    output = []
    output = []
    for item in tqdm(data):
        if item['acc'] == 0:
            continue
        question = item['question']
        solution = 'Lets think step by step.\n\nStep 1:' + item['solution']
        solution = split_steps(solution)
        revised_step = {}
        action = "Lets look at what we have now"
        action = "Alternative, we can"
        action = "Wait, lets "
        action = "Wait, lets break down this step"
        action = "Hm..."
        
        for i in range(1, len(solution) - 1):
            step = solution[i]
            prompt = ("Please modify the following math reasoning step to make it to be **incorrect**. You can consider add some calculation error. Just directly modify the content to make this step become incorrect, DO NOT add new sentence. Make the mistake looks reasonable.\n\n"
                      f"{step}")
            prefix = f"Original step: {step}\n\nRevised step:"
            out = rollout.generate(prompt, prefix)
            revised_step[i] = out
            # print(step)
            # print(out)
            # input()
        item['solution'] = solution
        item['revised'] = revised_step
        output.append(item)


def re_format():
    data = write_jsonl(output, 'data/Llama-1B-greedy-revised.jsonl')
    for item in data:
        [k, item['revised'].items()]
        item['prefix']


def entail(data, seed):
    # messages = [{'role': 'user', 'content': 'List three contries.'}]
    # response = run_azure(messages, model='gpt-4o-mini')
    # print(response)

    # model_name = 'meta-llama/Llama-3.2-3B-Instruct'
    # data = read_jsonl('data/Llama-1B-Branch.jsonl')
    # model_name = 'models/Llama-3.1-8B-Instruct'
    model_name = 'models/Qwen2.5-7B-Instruct'
    gpu_id = seed % 8
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        token = "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"
    )
    model = model.to(f'cuda:{gpu_id}')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # data = read_jsonl('data/Llama-1B-greedy.jsonl')
    rollout = RollOut(model, tokenizer, num=1, name='qwen')
    num = 0
    new_data = []
    for item in tqdm(data):
        correct = list(set([x[0] for x in item['correct']]))
        incorrect = list(set([x[0] for x in item['incorrect']]))
        nli = {}
        for x in correct:
            nli[x] = {}
            for y in incorrect:
                prompt = f'When solve a math problem, there are multiple directions for each step. Here are two steps (A and B) for the same prefix solution, they can explore the same direction with same or opposite results, or. Tell me whether the two step is:\n(1) Entailment: they explore the same direction and the results are not conflict\n(2) Contradiction: same direction but the two steps get different or conflict results, or explore the same direction but get irrelevant results.\n(3) Neutral: they explore different and irrelevant directions.\n\nA: {x}\n\nB: {y}\n\nOutput Entailment, Contradiction, or Neutral.'
                response = rollout.generate(prompt, prefix="After carefully checking the step A and step B, I can conclude that the answer must be:", add_nn=False, max_new_tokens=8)
                if 'Entail' in response:
                    label = 'Entailment'
                if 'Neutral' in response:
                    label = 'Neutral'
                if 'Contradict' in response:
                    label = 'Contradiction'
                    num += 1
                if 'Contradict' in response and seed == 0:
                    print(x)
                    print(y)
                    print(response)
                    print(num)
                # print(label)
                nli[x][y] = label
        item['nli'] = nli
        new_data.append(item)
    # write_jsonl(new_data, 'data/Llama-1B-Branch-NLI.jsonl')
    return new_data


def main():
    data = read_jsonl('data/Llama-1B-Branch.jsonl')
    # output = mp(pred, data, processes=16)
    output = mp(entail, data, processes=16)
    write_jsonl(output, 'data/Llama-1B-Branch-NLI.jsonl')

if __name__ == '__main__':
    main()




