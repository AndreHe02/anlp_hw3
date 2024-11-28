from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from transformers import pipeline
from datasets import load_dataset
import torch
from file_io import *
from tqdm import tqdm
from collections import Counter
from norm import math_normalizer as math_norm
from norm import numbered
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from collections import defaultdict
from scipy.stats import hypergeom
import random
import math
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


def get_weight_maj(answers, weights):
    answer_score = defaultdict(int)
    for a, s in zip(answers, weights):
        if len(a.strip()) > 0:
            answer_score[a] += s
    answer_score = [[s, a] for a, s in answer_score.items()]
    answer_score.sort(key=lambda x: x[0])
    if len(answer_score) == 0:
        answer = answers[0]
    else:
        answer = answer_score[-1][1]
    return answer


def real_try(N_answers, label, K):
    acc = []
    for N in range(1000):
        random.shuffle(N_answers)
        sub = N_answers[:K]
        acc.append(int(get_majority_vote(sub) == label))
    return sum(acc) / len(acc)


def real_pass(N_answers, label, K):
    acc = []
    for N in range(1000):
        random.shuffle(N_answers)
        sub = N_answers[:K]
        acc.append(int(label in sub))
    return sum(acc) / len(acc)


def real_weight_try(N_answers, label, K, weights):
    acc = []
    index = [i for i in range(len(N_answers))]
    for N in range(1000):
        random.shuffle(index)
        sub = [N_answers[i] for i in index[:K]]
        sub_w = [weights[i] for i in index[:K]]
        acc.append(int(get_weight_maj(sub, sub_w) == label))
    return sum(acc) / len(acc)


def real_bon(N_answers, label, K, weights):
    acc = []
    index = [i for i in range(len(N_answers))]
    for N in range(1000):
        random.shuffle(index)
        sub = [N_answers[i] for i in index[:K]]
        sub_w = [weights[i] for i in index[:K]]
        selected_ans = sub[np.argmax(sub_w)]
        acc.append(int(selected_ans == label))
    return sum(acc) / len(acc)


def wmv(file_name='out/gsm8k-64-branch-all.jsonl'):
    model_name = 'models/Llama-3.2-1B-Instruct-ORM'
    good_token_id = 15571  # Good (First token)
    bad_token_id = 17519  # Bad
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, torch_dtype=torch.float16, num_labels=1)
    # model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    model.config.pad_token_id = 128004
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = 128004
    model = model.cuda()
    model.eval()
    os.system('sh kill_gpu.sh')

    data = read_jsonl(file_name)
    bar = tqdm(data)
    acc = defaultdict(list)
    for item in bar:
        question = item['question']
        label = item['answer']
        label = numbered(label)
        outputs = item['solution']
        outputs = outputs[:64]

        answers = [math_norm(x) for x in outputs]

        for k in [1, 4, 10, 16, 32, 64]:
            acc[f"Maj@{k}"].append(real_try(answers, str(label), k))
        for k in [1, 4, 10, 16, 32, 64]:
            acc[f"Pass@{k}"].append(real_pass(answers, str(label), k))

        continue

        input_ids = []
        index = []
        for so in outputs:
            # so = "Lets think step by step.\n\nStep 1:" + so.replace('<|eot_id|>', '')
            # so = split_steps(so)
            # so = "\n\n".join([s.strip() + '<|reserved_special_token_0|>' for s in so])
            messages = [{'role': 'user', 'content': question}, {'role': 'assistant', 'content': so}]
            input_ids.append(
                torch.tensor(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)))
            index.append(len(input_ids[-1]) - 1)

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        input_ids = input_ids.cuda()
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=input_ids.ne(128004)).logits
            scores = logits.squeeze(-1).cpu().tolist()
            # scores = F.sigmoid(logits.view(-1)).cpu().tolist()
            # print(scores)
            # scores = F.sigmoid(logits.view(-1)).cpu().tolist()
            # scores = []
            # logits = logits[:, :, [good_token_id, bad_token_id]]
            # batch_scores = logits.softmax(dim=-1)[:, :, 0]
            # for i in range(batch_scores.size(0)):
            #     s = batch_scores[i, index[i]].cpu().tolist()
            #     # print(s)
            #     scores.append(s)

        for k in [1, 4, 10, 16, 32, 64]:
            acc[f"W-Maj@{k}"].append(real_weight_try(answers, str(label), k, scores))
        for k in [1, 4, 10, 16, 32, 64]:
            acc[f"BON@{k}"].append(real_bon(answers, str(label), k, scores))
        for k in [1, 4, 10, 16, 32, 64]:
            acc[f"W-Pass@{k}"].append(real_pass(answers, str(label), k))

    for k, v in acc.items():
        print(k, sum(v) / len(v))


# print(sum(acc) /len(acc))


def predict(data, seed, demos=None, model_name='meta-llama/Llama-3.2-1B-Instruct', num_sample=1):
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

        prompt = problem

        # prefix = item['solution'][:item['mistake']]
        # prefix = "\n\n".join(prefix)

        # if math_norm(item['solution'][-1]) == str(label):
        #     out_responses.append({'acc': 1, 'question': problem, 'answer': item['answer'], 'solution': ["\n\n".join(item['solution'])]})
        #     continue

        # print(prefix)
        # print('#' * 20)
        # print(item['correctness'])
        # boss_rollout = item['rollout_mem'][str(item['mistake'] - 1)]
        # boss_rollout = [x for x in boss_rollout if math_norm(x) == str(label)]
        # print(len(boss_rollout))
        # print(boss_rollout[0].replace('<|eot_id|>', ''))
        # print('#' * 20)
        # messages = [{"role": "user", "content": prompt}, {'role': 'assistant', 'content': prefix}]

        # messages = [{"role": "user", "content": prompt}, {'role': 'assistant', 'content': prefix + f'\n\nWait, this seems to be incorrect. Lets correct it.'}]
        # messages = [{"role": "user", "content": prompt}, {'role': 'assistant', 'content': prefix}]

        # input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        # input_ids = input_ids[:-1] + [271]

        # messages = [{"role": "user", "content": prompt}, {'role': 'assistant', 'content': item['initial_reason_steps'] + item['rejected'] + '\nWait, this step is incorrect! Lets try again:'}]

        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)

        # messages = [{"role": "user", "content": prompt}, {'role': 'assistant', 'content': "Lets think step by step.\n\nStep 1:"}]
        # input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        # input_ids = input_ids[:-1]

        out = model.generate(
            input_ids=torch.tensor([input_ids]).to(f'cuda:{gpu_id}'),
            max_new_tokens=1024,
            do_sample=num_sample != 1,
            temperature=1,
            top_p=0.9,
            num_return_sequences=num_sample,
            pad_token_id=tokenizer.eos_token_id,
        )

        # prefix = [tokenizer.decode(x).replace('<|eot_id|>', '').strip() for x in out]
        # prefix = [tokenizer.encode(x + '\n\nIn conclusion, the final answer is: $\\boxed{') for x in prefix]
        # out = []
        # for p in prefix:
        #     out.append(model.generate(
        #         input_ids=torch.tensor([p]).to(f'cuda:{gpu_id}'), 
        #         max_new_tokens=128,
        #         do_sample=False,
        #         temperature=1.0,
        #         top_p=1,
        #         pad_token_id=tokenizer.eos_token_id,
        #     )[0])

        # out = [x[len(input_ids):] for x in out]
        # outputs = tokenizer.decode(out)
        outputs = [tokenizer.decode(x).replace('<|eot_id|>', '') for x in out]
        # outputs = outputs[0]["generated_text"][-1]['content']
        print(outputs[0])

        answers = [math_norm(x) for x in outputs]
        answer = get_majority_vote(answers)
        print(answer, label)
        # answer = answers[0]
        acc.append(int(str(answer) == str(label)))
        out_responses.append({'acc': acc[-1], 'question': problem, 'answer': item['answer'], 'solution': outputs})
        bar.set_postfix(acc=sum(acc) / len(acc))
    model = model.cpu()
    del model
    torch.cuda.empty_cache()
    return out_responses


def main():
    from prompt import MATH_LEVEL2, GSM8K
    # model_name = 'meta-llama/Llama-3.2-3B-Instruct'
    # model_name = 'models/Llama-3.1-8B-Instruct'
    model_name = 'models/Llama-3.2-1B-Instruct-PPO/policy'

    # data = load_dataset('AI-MO/aimo-validation-math-level-4')['train']
    # data = load_dataset('openai/gsm8k', 'main')['test']
    # data = load_dataset('xinlai/Math-Step-DPO-10K')['train']
    data = load_dataset('openai/gsm8k', 'main')['test']

    # data = read_jsonl('out/gsm8k-32-1b-short-rollout.jsonl')

    # data = read_jsonl('data/test.jsonl')
    # data = read_jsonl('data/train_math_gsm.jsonl')

    # data = [x for x in data][100:]

    data = [x for x in data][:100]
    # data = [x for x in data]
    # output = predict(data, 0, num_sample=1, demos=MATH_LEVEL2, model_name=model_name)
    output = mp(predict, data, processes=16, num_sample=32, demos=MATH_LEVEL2, model_name=model_name)
    # write_jsonl(output, 'out/ckpt-gsm8k-32-1b-nli.jsonl')
    write_jsonl(output, 'out/gsm8k-ppo-32-1b.jsonl')
    # write_jsonl(output, 'data/Llama-3B-Branch.jsonl')

    acc = [x['acc'] for x in output]
    print()
    print("Final Acc", sum(acc) / len(acc))

    write_jsonl(output, 'out/tmp.jsonl')


if __name__ == '__main__':
    main()
    wmv('out/gsm8k-ppo-32-1b.jsonl')
    # main()


























