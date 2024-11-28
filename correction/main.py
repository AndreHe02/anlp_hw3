import os
import re
import signal
import subprocess
import tempfile
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import pandas as pd
from datasets import load_dataset, Dataset, concatenate_datasets
import torch
from transformers import set_seed
from tqdm import tqdm
from file_io import *
from openai import OpenAI
# from vllm import LLM, SamplingParams

import sglang as sgl

sgl.gen()

import os

os.environ['http_proxy'] = 'http://agent.baidu.com:8891'
os.environ['https_proxy'] = 'http://agent.baidu.com:8891'


@dataclass
class Config:
    model_id: str

    # Decoding Parameters
    num_samples: int  # Number of candidates to generate (width)
    num_generations: int  # Number of steps to generate per candidate (depth)
    restart_on_fail: bool  # Regenerate a step if it fails to generate Python codeblocks

    # Sampling Parameters
    temperature: float
    max_new_tokens: int

    # Runtime Parameters
    validation_set: str  # One of AI-MO/aimo-validation-amc, AI-MO/aimo-validation-aime, AI-MO/aimo-validation-math-level-4, AI-MO/aimo-validation-math-level-5
    is_submission: bool = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))


def build_vllm(config):
    # num_gpus = torch.cuda.device_count()
    # if "awq" in config.model_id.lower():
    #     quantization = "AWQ"
    # elif "gptq" in config.model_id.lower():
    #     quantization = "gptq"
    # else:
    #     quantization = None
    # vllm = LLM(
    #     model=config.model_id,
    #     tensor_parallel_size=num_gpus,
    #     quantization=quantization,
    #     swap_space=0,
    # )
    pipe = pipeline("text-generation", model=config.model_id, torch_dtype=torch.bfloat16, device='cuda')
    return pipe


def apply_template(sample, tokenizer, prompt):
    messages = [{"role": "user", "content": prompt.format(sample["prompt"], "{}")}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    sample["text"] = text
    return sample


def generate_batched(samples, pipe, sampling_params):
    outputs = pipe(samples["gen_texts"], **sampling_params)
    # outputs[0]["generated_text"]
    # samples["gen_texts"] = [o.prompt + o.outputs[0].text for o in outputs]
    samples["gen_texts"] = [o[0]["generated_text"] for o in outputs]
    return samples

@sgl.function
def generate_sgl(s, question, sampling_params):
    s += "Q: " + question + "\n"
    s += "A:" + sgl.gen("answer", stop="\n",**sampling_params)


class PythonREPL:
    def __init__(self, timeout=5):
        self.timeout = timeout

    @contextmanager
    def time_limit(self, seconds):
        def signal_handler(*_):
            raise TimeoutError(f"Timed out after {seconds} seconds.")

        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)

    def __call__(self, query):
        query = "import math\nimport numpy as np\nimport sympy as sp\n" + query
        query = query.strip().split("\n")
        if "print(" not in query[-1]:
            if "#" in query[-1]:
                query[-1] = query[-1].split("#")[0]
            query[-1] = "print(" + query[-1] + ")"
        query = "\n".join(query)
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, "tmp.py")
            with open(temp_file_path, "w", encoding="utf-8") as f:
                f.write(query)
            with self.time_limit(self.timeout):
                result = subprocess.run(
                    ["python3", temp_file_path],
                    capture_output=True,
                    check=False,
                    text=True,
                    timeout=self.timeout,
                )
                if result.returncode == 0:
                    output = result.stdout
                    return True, output.strip()
                error_msg = result.stderr.strip()
                msgs = error_msg.split("\n")
                new_msgs = []
                want_next = False
                for m in msgs:
                    if "Traceback" in m:
                        new_msgs.append(m)
                    elif m == msgs[-1]:
                        new_msgs.append(m)
                    elif temp_file_path in m:
                        st = m.index('"/') + 1 if '"/' in m else 0
                        ed = m.index(temp_file_path) + 1 if temp_file_path in m else None
                        clr = m[st:ed] if not ed else m[st:]
                        m = m.replace(clr, "")
                        new_msgs.append(m)
                        want_next = True
                    elif want_next:
                        new_msgs.append(m)
                        want_next = False
                error_msg = "\n".join(new_msgs)
                return False, error_msg.strip()


def execute_completion(executor, completion, return_status, last_code_block):
    executions = re.findall(r"```python(.*?)```", completion, re.DOTALL)
    if len(executions) == 0:
        return completion, False if return_status else completion
    if last_code_block:
        executions = [executions[-1]]
    outputs = []
    successes = []
    for code in executions:
        success = False
        for lib in ("subprocess", "venv"):
            if lib in code:
                output = f"{lib} is not allowed"
                outputs.append(output)
                successes.append(success)
                continue
        try:
            success, output = executor(code)
        except TimeoutError as e:
            print("Code timed out")
            output = e
        if not success and not return_status:
            output = ""
        outputs.append(output)
        successes.append(success)
    output = str(outputs[-1]).strip()
    success = successes[-1]
    if return_status:
        return output, success
    return output


def postprocess_completion(text, return_status, last_code_block):
    executor = PythonREPL()
    result = execute_completion(executor, text, return_status=return_status, last_code_block=last_code_block)
    del executor
    return result


def extract_boxed_answer(text):
    def last_boxed_only_string(text):
        idx = text.rfind("\\boxed")
        if idx < 0:
            idx = text.rfind("\\fbox")
            if idx < 0:
                return None
        i = idx
        right_brace_idx = None
        num_left_braces_open = 0
        while i < len(text):
            if text[i] == "{":
                num_left_braces_open += 1
            if text[i] == "}":
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break
            i += 1
        if right_brace_idx is None:
            return None
        return text[idx: right_brace_idx + 1]

    def remove_boxed(boxed):
        left = "\\boxed{"
        try:
            assert boxed[: len(left)] == left
            assert boxed[-1] == "}"
            length = len(left)
            return boxed[length:-1]
        except Exception:
            return None

    boxed = last_boxed_only_string(text)
    if boxed is None:
        return None
    answer = remove_boxed(boxed)
    return answer


def normalize_answer(answer):
    match = re.search(r"(.*?)Problem:", answer, flags=re.S)
    if match:
        answer = match.group(1)
    subs = [("an ", ""), ("a ", ""), (".$", "$"), ("\\$", ""), (r"\ ", ""), (" ", ""), ("mbox", "text"),
            (",\\text{and}", ","), ("\\text{and}", ","), ("\\text{m}", "\\text{}"), ("\\le", "<")]
    remove = ["square", "ways", "integers", "dollars", "mph", "inches", "ft", "hours", "km", "units", "\\ldots", "sue",
              "points", "feet", "minutes", "digits", "cents", "degrees", "cm", "gm", "pounds", "meters", "meals",
              "edges", "students", "childrentickets", "multiples", "\\text{s}", "\\text{.}", "\\text{\ns}",
              "\\text{}^2", "\\text{}^3", "\\text{\n}", "\\text{}", r"\mathrm{th}", r"^\circ", r"^{\circ}", r"\;",
              r",\!", "{,}", '"', "\\dots", "\n", "\r", "\f", "\%"]
    sub_patterns = [r"(\\text\{)(.*?)(\})", r"(\\textbf\{)(.*?)(\})", r"(\\overline\{)(.*?)(\})",
                    r"(\\boxed\{)(.*)(\})"]
    split_patterns = [r"finalansweris(.*)", r"answer?is:?(.*)", r"oxed\{(.*?)\}", r"\$(.*?)\$"]
    for before, after in subs:
        answer = answer.replace(before, after)
    for expr in remove:
        answer = answer.replace(expr, "")
    for pattern in sub_patterns:
        answer = re.sub(pattern, "\\2", answer)
    for pattern in split_patterns:
        if len(re.findall(pattern, answer)) > 0:
            answer = re.findall(pattern, answer)[-1]
    answer = answer.strip()
    if "rac" in answer and "\\frac" not in answer:
        answer = answer.replace("rac", "\\frac")
    answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", answer)
    answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", answer)
    answer = answer.replace("$", "")
    if answer.replace(",", "").isdigit():
        answer = answer.replace(",", "")
    return answer


def process_code(sample, restart_on_fail, last_step, check_last_n_chars=100):
    gen_text = sample["gen_texts"]
    num_python_blocks = len(re.findall(r"```python(.*?)```", gen_text, re.DOTALL))
    region_to_check = gen_text[-check_last_n_chars:]
    if num_python_blocks == 0:
        if restart_on_fail:
            print("no code has ever been generated, RESTARTING")
            sample["gen_texts"] = sample["text"]
        else:
            print("no code has ever been generated, STOP")
            sample["should_prune"] = True
            sample["has_code"] = False
        return sample
    if not gen_text.endswith("```output\n") and ("answer is" in region_to_check or "\\boxed" in region_to_check):
        num_output_blocks = len(re.findall(r"```output(.*?)```", gen_text, re.DOTALL))
        if num_output_blocks == 0:
            print("The model hallucinated the code answer")
            sample["should_prune"] = True
            return sample
        if "boxed" in region_to_check:
            try:
                answer = normalize_answer(extract_boxed_answer(region_to_check))
            except Exception:
                answer = "-1"
        else:
            answer = normalize_answer(region_to_check)
        sample["model_answers"] = answer
        return sample
    if last_step:
        return sample
    if not gen_text.endswith("```output\n"):
        print("warning: output block not found: ", gen_text[-40:])
        if restart_on_fail:
            sample["gen_texts"] = sample["text"]
        else:
            sample["should_prune"] = True
        return sample
    code_result, _ = postprocess_completion(gen_text, return_status=True, last_code_block=True)
    truncation_limit = 200
    if len(code_result) > truncation_limit:
        code_result = code_result[:truncation_limit] + " ... (output truncated)"
    sample["gen_texts"] = gen_text + f"{code_result}\n```"
    return sample


def filter_answers(answers):
    def validate_answer_is_numeric(x, tolerance=0.2):
        try:
            x = round(float(x))
            f = float(x)
            if abs(x - f) > tolerance:
                x = -1
        except Exception:
            x = -1
        return x

    formatted = [validate_answer_is_numeric(a) for a in answers]
    return formatted


def get_majority_vote(answers):
    if not len(answers):
        return 0
    c = Counter(answers)
    value, _ = c.most_common()[0]
    return value


def run_gpt(messages, model="gpt-4o", **kwargs):
    client = OpenAI(api_key='sk-proj-wdadANwH9UDrWT6ggTCtT3BlbkFJtYmQznI2IeoVi1xXuFnR')
    completion = client.chat.completions.create(model=model, messages=messages, **kwargs)
    return [completion.choices[i].message.content for i in range(len(completion.choices))]


def main(config):
    print(f"=== Running submission with config ===\n\n{config}")
    set_seed(42)
    num_procs = os.cpu_count()
    pipe = build_vllm(config)
    sampling_params = dict(
        temperature=config.temperature,
        max_tokens=config.max_new_tokens,
        stop=["```output\n"],
        # stop_strings=["```output\n"],
        # do_sample=True,
        # tokenizer=pipe.tokenizer
        # include_stop_str_in_output=True,
    )

    save_path = f"out/amc-NuminaMath-t{config.temperature}-n{config.num_samples}"

    # env, iter_test = get_kaggle_env(config)
    test_data = load_dataset('AI-MO/aimo-validation-amc')['train']
    final_answers = []
    acc = []
    log = []
    os.system('sh kill_gpu.sh')
    for item in tqdm(test_data, total=len(test_data), desc="Solving problems"):
        problem = item['problem']
        data_id = item['id']
        chat_problem = apply_template({"prompt": problem}, tokenizer=pipe.tokenizer, prompt="{}")
        print(save_path)
        print(f"=== INPUT FOR PROBLEM ID {data_id} ===\n{problem}\n")
        samples = Dataset.from_list([
            {
                "text": chat_problem["text"],
                "gen_texts": chat_problem["text"],
                "should_prune": False,
                "model_answers": "-1",
                "has_code": True,
            }
            for _ in range(config.num_samples)
        ])
        completed = []
        for step in range(config.num_generations):
            samples = samples.map(
                generate_batched,
                batch_size=128,
                batched=True,
                fn_kwargs={"pipe": pipe, "sampling_params": sampling_params},
                load_from_cache_file=False,
            )
            samples = samples.map(
                process_code,
                num_proc=num_procs,
                load_from_cache_file=False,
                fn_kwargs={"restart_on_fail": config.restart_on_fail,
                           "last_step": step == (config.num_generations - 1)},
            )
            done = samples.filter(lambda x: x["should_prune"] is True, load_from_cache_file=False)
            if len(done):
                completed.append(done)
            samples = samples.filter(lambda x: x["should_prune"] is False, load_from_cache_file=False)
        completed.append(samples)
        samples = concatenate_datasets(completed)
        candidates = samples["model_answers"]
        print(f"=== CANDIDATE ANSWERS ({len(candidates)}) ===\n{candidates}\n")
        filtered = filter_answers(candidates)
        print(f"=== FILTERED ANSWERS ({len(filtered)}) ===\n{filtered}\n")
        majority = get_majority_vote(filtered)
        print(f"=== MAJORITY ANSWER (mod 1000) ===\n{majority}\n")
        predicted_answer = int(majority)
        ground_truth = int(item['answer'])
        print(predicted_answer)
        print(ground_truth)
        acc.append(int(predicted_answer == ground_truth))
        print('ACC', sum(acc) / len(acc))
        print(samples)
        log.append({
            'id': data_id,
            'problem': problem,
            'model': config.model_id,
            'temperature': config.temperature,
            'n': config.num_samples,
            'samples': samples.to_list(),
            'candidates': candidates,
            'majority': majority,
            'label': ground_truth
        })
        write_json(log, save_path)
        if len(log) >= 100:
            break
    print(sum(acc) / len(acc))
    print(sum(acc))


if __name__ == '__main__':
    config = Config(
        model_id="AI-MO/NuminaMath-7B-TIR",
        num_samples=1,
        num_generations=4,
        restart_on_fail=True,
        temperature=1,
        max_new_tokens=2048,
        validation_set="AI-MO/aimo-validation-amc",
    )
    main(config)
