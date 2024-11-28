from datasets import load_dataset
from transformers import AutoTokenizer
from file_io import *
from tqdm import tqdm
from collections import Counter
from norm import math_normalizer as math_norm
from norm import numbered
from collections import defaultdict
import random
import re
import os


def build_gold():
    data = read_jsonl('data/phase2_train.jsonl')
    print(len(data))
    new_data = []
    for item in tqdm(data):
        question = item['question']['problem']
        answer = item['question']['ground_truth_answer']
        solution = []
        if item['label']['finish_reason'] != 'solution':
            continue
        for step in item['label']['steps']:
            chosen_completion = step["human_completion"] if step["chosen_completion"] is None else \
                step["completions"][step["chosen_completion"]]["text"]
            if chosen_completion is None:
                chosen_completion = step["completions"][0]["text"]
            solution.append(chosen_completion)
        new_data.append({'question': question, 'answer': answer, 'solution': solution})
    write_jsonl(new_data, 'data/PRM_phase2_solution.jsonl')


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


def get_majority_vote(answers):
    answers = [a for a in answers if len(a.strip()) > 0]
    if len(answers) == 0:
        return ''
    c = Counter(answers)
    value, _ = c.most_common()[0]
    return value

class SGGenerate:
    def __init__(self):
        import openai
        self.tokenizer = AutoTokenizer.from_pretrained('models/Llama-3.2-3B-Instruct')
        self.client = openai.Client(
            base_url="http://127.0.0.1:30000/v1", api_key="EMPTY")

    def generate(self, messages, n=1):
        if messages[-1]['role'] == 'user':
            response = self.client.chat.completions.create(
                model="default",
                messages=messages,
                temperature=1,
                max_tokens=2048,
                top_p=0.9,
                n=n
            )
            response = [x.message.content for x in response.choices]
        else:
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            prompt = prompt[:-10]
            response = self.client.completions.create(
                model="default",
                prompt=prompt,
                temperature=1,
                max_tokens=2048,
                top_p=0.9,
                n=n
            )
            response = [x.text for x in response.choices]
        return response


def predict(data, seed, num_sample=1):
    gpu_id = seed % 8
    model = SGGenerate()

    os.system('sh kill_gpu.sh')
    bar = tqdm(data)
    acc = []
    out_responses = []
    for item in bar:
        for name in ['prompt', 'problem', 'question']:
            if name in item:
                problem = item[name]
                break
        else:
            problem = None
        label = item['answer']
        if isinstance(label, str):
            if '####' in label:
                label = numbered(label)
            else:
                label = math_norm("\\boxed{" + label + '}')
        # prompt = f"Instruction: Answer the math questions step by step and mark the answer (a number) in \\boxed{{}}.\n\n{demos}\n\nQuestion: {problem}"
        prompt = problem

        messages = [{"role": "user", "content": prompt}, {'role': 'assistant', 'content': "Lets think step by step.\n\nStep 1:"}]

        outputs = model.generate(
            messages,
            n=num_sample,
        )
        print(outputs[0])

        answers = [math_norm(x) for x in outputs]
        answer = get_majority_vote(answers)
        print(answer, label)
        # answer = answers[0]
        acc.append(int(str(answer) == str(label)))
        out_responses.append({'acc': acc[-1], 'question': problem, 'answer': item['answer'], 'solution': outputs})
        bar.set_postfix(acc=sum(acc) / len(acc))
    return out_responses


def main():
    # model_name = 'meta-llama/Llama-3.2-3B-Instruct'
    model_name = 'models/Llama-3.2-1B-Instruct-Branch-Cal-ALl'

    # data = load_dataset('AI-MO/aimo-validation-math-level-4')['train']
    # data = load_dataset('openai/gsm8k', 'main')['test']
    # data = load_dataset('xinlai/Math-Step-DPO-10K')['train']
    data = load_dataset('openai/gsm8k', 'main')['test']
    # data = read_jsonl('data/test.jsonl')
    # data = read_jsonl('data/train_math_gsm.jsonl')

    # data = [x for x in data][100:]
    data = [x for x in data][:100]
    # data = [x for x in data]
    output = mp(predict, data, processes=16, num_sample=5)
    write_jsonl(output, 'out/gsm8k-64-branch-cal-all.jsonl')
    # write_jsonl(output, 'data/Llama-1B-long-sample10.jsonl')

    acc = [x['acc'] for x in output]
    print()
    print("Final Acc", sum(acc) / len(acc))

    write_jsonl(output, 'out/tmp.jsonl')


if __name__ == '__main__':
    wmv('out/gsm8k-64-branch-cal-all.jsonl')
    # main()


























