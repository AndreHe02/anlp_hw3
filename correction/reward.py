from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from datasets import load_dataset
import torch
from file_io import *
from tqdm import tqdm
from collections import Counter
from norm import math_normalizer as math_norm
from norm import get_majority_vote, numbered
import numpy as np
import random
import re
import os


def passn():
    data = read_jsonl('data/Llama-1B-sample10-2-new.jsonl')
    model_name = 'models/Llama-3.2-3B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(len(data))
    acc = []
    recall = []
    length = []
    max_len = []
    min_len = []
    max_data = []
    min_data = []
    filter_data = []
    process_wrong = []
    gold_data = []
    orm = []
    null_answer = 0
    for item in tqdm(data):
        solutions = item['solution']
        solutions = [a.replace('<|eot_id|>', '').strip() for a in solutions]
        answers = [math_norm(x) for x in solutions]
        # ll = [len(tokenizer.encode(x)) for x in solutions]
        ll = [0 for x in solutions]
        null_answer += len([a for a in solutions if "In conclusion, the final answer is:" not in a])
        answer = get_majority_vote(answers)
        label = item['answer']

        true_answer = [[li, s] for li, a, s in zip(ll, answers, solutions) if str(a) == str(label)]
        true_answer.sort(key=lambda x: x[0])
        if len(true_answer) > 0:
            gold_data.append({'question': item['question'], 'answer': label, 'solution': [x[1] for x in true_answer]})
        wrong_answer = [s for a, s in zip(answers, solutions) if str(a) != str(label)]
        wrong_answer = [a for a in wrong_answer if "In conclusion, the final answer is:" in a]
        if len(true_answer) >= 1 and len(wrong_answer) >= 1:
            process_wrong.append(
                {'question': item['question'], 'answer': label, 'solution': wrong_answer, 'all_solution': solutions})

        # wrong_answer = [a for a in wrong_answer if "In conclusion, the final answer is:" in a]
        if len(true_answer) >= 1 and len(wrong_answer) >= 1:
            orm.append({'question': item['question'], 'answer': label, 'good_solution': [a[1] for a in true_answer], 'bad_solution': wrong_answer})
        
        if len(true_answer) >= 1 and len(wrong_answer) >= 1:
            filter_data.append(item)

        if len(true_answer) > 1:
            max_len.append(true_answer[-1][0])
            min_len.append(true_answer[0][0])
            max_data.append({'question': item['question'], 'answer': label, 'solution': true_answer[-1][1]})
            min_data.append({'question': item['question'], 'answer': label, 'solution': true_answer[0][1]})

        acc.append(int(str(answer) == str(label)))
        recall.append(int(str(label) in answers))

        length.extend(ll)

    # write_jsonl(orm, 'data/Llama-1B-ORM.jsonl')
    # print(len(orm))
    print(len(filter_data))
    write_jsonl(filter_data, 'data/Llama-1B-sample10-2-new-filter.jsonl')
    # write_jsonl(process_wrong, 'data/Llama-3B-sample5-2-process-wrong.jsonl')
    # print(len(process_wrong))
    # write_jsonl(max_data, 'data/Llama-1B-long-sample10-max-true.jsonl')
    # write_jsonl(min_data, 'data/Llama-1B-long-sample10-min-true.jsonl')
    # print(null_answer)
    # print(sum(acc) / len(acc))
    # print(sum(recall) / len(recall))
    # print('max', sum(max_len) / len(max_len), len(max_len))
    # print('min', sum(min_len) / len(min_len), len(min_len))
    # print(max(length))
    # print(sum(length) / len(length))
    # print(len([x for x in length if x > 400]))


def build_path():
    data = read_jsonl('data/Llama-3B-sample5-2-process-wrong-rollout.jsonl')
    print(len(data))
    new_data = []
    journey = []
    for item in tqdm(data):
        question = item['question']
        solution = item['solution']
        mistake = item['mistake']
        rollout = item['rollout_mem']
        item['answer'] = str(item['answer'])

        if mistake == 0:
            mistake = 1
            if '0' not in rollout:
                rollout['0'] = []
            rollout['0'].extend(item['all_solution'])

        prefix = solution[:mistake]
        incorrect = [solution[mistake:]]
        if str(mistake) in rollout:
            for cand in rollout[str(mistake)]:
                if math_norm(cand) != item['answer'] and 'final answer' in cand:
                    incorrect.append([solution[mistake]] + split_steps(cand))

        correct = []
        for cand in rollout[str(mistake - 1)]:
            if math_norm(cand) == item['answer'] and 'final answer' in cand:
                if len(split_steps(cand)) == 0:
                    print(cand.replace('<|eot_id|>' ,''))
                    input()
                correct.append(split_steps(cand))
        
        value_prefix = [int(math_norm(s) == item['answer']) for s in rollout[str(mistake - 1)]]
        value_prefix = sum(value_prefix) / len(value_prefix)

        # value_next = [int(math_norm(s) == item['answer']) for s in rollout[str(mistake)]]
        # value_next = sum(value_next) / len(value_next)

        if len(correct) == 0 or len(incorrect) == 0:
            continue
        
        base_value = [int(math_norm(x) == item['answer']) for x in item['all_solution']]
        base_value = sum(base_value) / len(base_value)
        new_data.append({'question': question, 'prefix': prefix, 'incorrect': incorrect, 'correct': correct, 'prefix_value': value_prefix, 'base_vale': base_value})
        
        # for i in range(mistake - 1, 0, -1):
        #     if str(i) in rollout:
        #         deep_correct = [split_steps(cand) for cand in rollout[str(i)] if math_norm(cand) == item['answer']]
        #         deep_value = len(correct) / 10
        #         deep_incorrect = []
        # print('\n**********\n'.join(prefix))

        if False:
            question = new_data[-1]['question']
            prefix = new_data[-1]['prefix']
            correct = new_data[-1]['correct']
            incorrect = new_data[-1]['incorrect']
            prefix_value = new_data[-1]['prefix_value']
            base_value = new_data[-1]['base_vale']

            max_att = 1
            max_dep = 1

            seed = np.random.randn()
            correct: list = random.choice(correct)
            if False:
                answer = prefix + correct
            else:
                reject = random.choice(incorrect)
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
                if len(prefix) == 0:
                    print(prefix)
                    input()
                if len(reject) == 0:
                    print(reject)
                    input()
                if len(correct) == 0:
                    print(correct)
                    input()
                answer = prefix + reject + [f"Wait, I need to rethink my answer... Hmm..., there seem to be some issues. Lets check. Alright, let’s go back{back_to} and re-try {back_step} in a different way."] + correct

            answer = '\n\n'.join(answer)
            answer = "Let's solve this problem step by step, and I will reflect on each step and promptly backtrack and correct. " + answer

            print(answer)
            input()
            print('\n**********\n'.join(prefix))
            print('#' * 10 + 'Incorrect')
            print('\n**********\n'.join(incorrect[-1]))

            print('#' * 10 + 'Correct')
            print('\n**********\n'.join(correct[-1]))
            
            print(item['answer'])
            print(value_prefix)
            print(len(incorrect), len(correct))
            input()

    print(len(new_data))
    write_jsonl(new_data, 'data/New-Branch.jsonl')


def show_tree():
    data = read_jsonl('data/Llama-1B-sample10-2-process-wrong-rollout-new.jsonl')
    print(len(data))
    new_data = []
    journey = []
    for item in data:
        question = item['question']
        solution = item['solution']
        mistake = item['mistake']
        if mistake == 0:
            mistake = 1
        prompt = []
        label = []

        for i, step in enumerate(solution):
            prompt.append(step.strip() + '<|reserved_special_token_0|>')
            label.append(step.strip() + (' Good' if i < mistake else ' Bad'))
        new_data.append({'question': question, 'prompt': prompt, 'label': label})

        rollout = item['rollout_mem'][str(mistake - 1)]
        good_solutions = [s for s in rollout if math_norm(s) == item['answer']]
        if len(good_solutions) > 0:
            good_solution = good_solutions[0]
            good_solution = '\n\n'.join(solution[:mistake]) + good_solution.replace('<|eot_id|>', '')
            good_solution = split_steps(good_solution)
            for i, step in enumerate(solution):
                prompt.append(step.strip() + '<|reserved_special_token_0|>')
                label.append(step.strip() + ' Good')
            new_data.append({'question': question, 'prompt': prompt, 'label': label})
        if len(good_solutions) > 0:
            good_solution = good_solutions[0]

    print(len(new_data))
    write_jsonl(new_data, 'data/Llama-1B-PRM.jsonl')


def mp(func, data, processes=20, **kwargs):
    from torch.multiprocessing import multiprocessing
    import copy
    pool = multiprocessing.Pool(processes=processes)
    length = len(data) // processes + 1
    results = []
    for ids in range(processes):
        collect = data[ids * length:(ids + 1) * length]
        kwargs['seed'] = ids
        results.append(pool.apply_async(func, args=(collect,), kwds=copy.deepcopy(kwargs)))
    pool.close()
    pool.join()
    result_collect = []
    for j, res in enumerate(results):
        result = res.get()
        result_collect.extend(result)
    return result_collect


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


class RollOut:
    def __init__(self, model, tokenizer, num=10, question=None, answer=None):
        self.model = model
        self.tokenizer = tokenizer
        self.num = num
        self.question = question
        self.answer = answer
        self.target = None
        self.rollout_mem = dict()

    def rollout(self, input_ids):
        out = self.model.generate(
            input_ids=torch.tensor([input_ids]).to(self.model.device),
            max_new_tokens=1024,
            do_sample=True,
            temperature=1.0,
            top_p=0.9,
            num_return_sequences=self.num,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        out = [x[len(input_ids):] for x in out]
        outputs = [self.tokenizer.decode(x) for x in out]
        return outputs

    def is_correct(self, steps, step_id):
        messages = [{"role": "user", "content": self.question},
                    {"role": "assistant", "content": "\n\n".join(steps[:step_id + 1])}]
        input_ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        input_ids = input_ids[:-1] + [271]
        outputs = self.rollout(input_ids)
        self.rollout_mem[step_id] = outputs

        answers = [math_norm(x) for x in outputs]
        match = [a == str(self.answer) for a in answers]
        value = sum(match) / len(match)
        head = str(self.answer) in answers
        return value

    def fake_is_correct(self, steps, step_id):
        if self.target is None:
            self.target = np.random.randint(0, len(steps))
        return step_id < self.target


def bisearch(steps, is_correct):
    n = len(steps)
    left, right = 0, n - 1
    correctness = [None for _ in range(n)]
    while left < right:
        mid = (left + right) // 2
        value = is_correct(steps, mid)
        correctness[mid] = value
        if value:
            left = mid + 1  # Move to the right part
        else:
            right = mid  # Keep looking in the left part
    return left, correctness


def full_search(steps, is_correct):
    n = len(steps)
    left, right = 0, n - 1
    correctness = [None for _ in range(n)]
    mistake = n
    for i in range(1, n - 1):
        value = is_correct(steps, i)
        correctness[i] = value
        if value == 0:
            mistake = min(i, mistake)
    return left, correctness


def predict(data, seed, demos=None, model_name='meta-llama/Llama-3.2-1B-Instruct'):
    gpu_id = seed % 8
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    )
    model = model.to(f'cuda:{gpu_id}')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    os.system('sh kill_gpu.sh')
    bar = tqdm(data)
    acc = []
    out_responses = []
    start_time = time.time()
    for item in bar:
        for name in ['prompt', 'problem', 'question']:
            if name in item:
                problem = item[name]
                break
        else:
            problem = None
        label = item['answer']
        # label = str(label)
        label = str(numbered(label))
        item['all_solution'] = item['solution']
        prompt = problem
        # wrong = []
        # for solution in item['solution']:
        #     if 'the final answer' in solution and math_norm(solution) != str(label):
        #         wrong.append(solution)
        # candidate = item['solution'] if len(wrong) == 0 else wrong
        # solution = random.choice(candidate)

        correct_solution = [x for x in item['solution'] if math_norm(x) == str(label)]
        wrong_solution = [x for x in item['solution'] if math_norm(x) != str(label) and 'the final answer' in x]

        candidate = []
        # if len(correct_solution) > 0:
        #     candidate.append(random.choice(correct_solution))
        if len(wrong_solution) > 0:
            candidate.append(random.choice(wrong_solution))
        if len(correct_solution) > 0:
            candidate.append(random.choice(correct_solution))

        if len(wrong_solution) == 0:
            continue
        if len(correct_solution) == 0:
            continue

        for solution in candidate:
            solution = 'Lets think step by step.\n\nStep 1:' + solution
            solution = split_steps(solution)
            solution = ["Let's solve this problem step by step."] + solution
            agent = RollOut(model, tokenizer, num=30, question=prompt, answer=label)
            mistake, correctness = full_search(solution, agent.is_correct)
            rollout_mem = agent.rollout_mem
            correctness[0] = sum([int(math_norm(x) == str(label)) for x in item['solution']]) / len(item['solution'])
            correctness[-1] = float(math_norm(solution[-1]) == str(label))

            print(correctness, (time.time() - start_time) / (60 * 60 * 9))

            out_responses.append({
                'question': problem,
                'answer': item['answer'],
                'solution': solution,
                'mistake': mistake,
                'correctness': correctness,
                'rollout_mem': rollout_mem,
                'all_solution': item['all_solution'],
            })
        time_cost = time.time() - start_time
        if time_cost > 60 * 60 * 9:
            break

    return out_responses


def main():
    from prompt import MATH_LEVEL2, GSM8K
    # model_name = 'meta-llama/Llama-3.2-1B-Instruct'
    model_name = 'models/Llama-3.2-1B-Instruct-SFT'
    # model_name = 'models/Llama-3.2-3B-Instruct-Branch-Shortcut'
    # model_name = 'models/Llama-3.2-1B-Instruct-Branch-Short'

    # data = read_jsonl('data/Llama-1B-sample10-2-new.jsonl')
    # print(len(data))
    # da = read_jsonl('data/Llama-1B-sample10-2-new-rollout-all.jsonl')
    # da_q = set([x['question'] for x in da])
    # data = [x for x in data if x['question'] not in da_q]

    data= read_jsonl('out/gsm8k-32-1b.jsonl')
    print(len(data))
    # data = read_jsonl('out/gsm8k-32-1b-short.jsonl')
    # data = [x for x in data if x['acc'] == 0]
    # data = [x for x in data][100:]
    # data = [x for x in data][:100]
    output = mp(predict, data, processes=24, model_name=model_name)
    # write_jsonl(output, 'data/Llama-1B-sample10-2-new-rollout-all-part2.jsonl')
    write_jsonl(output, 'data/gsm8k-1b-Value.jsonl')

    # write_jsonl(output, 'out/gsm8k-32-1b-short-rollout.jsonl')
    print()
    # print("Final Acc", sum(acc) / len(acc))


if __name__ == '__main__':
    passn()
    # build_path()
    # main()
    # data = read_jsonl('data/Llama-1B-sample10-2-new-rollout-all.jsonl') + read_jsonl('data/Llama-1B-sample10-2-new-rollout-all-part2.jsonl')
    # write_jsonl(data, 'data/Llama-1B-Value-Neg.jsonl')
    # da = read_jsonl('data/Llama-1B-sample10-2-new-rollout-all.jsonl')
    # print(len(da))










