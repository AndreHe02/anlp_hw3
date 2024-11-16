from datasets import load_dataset
from tot.tasks.base import Task
from tot.prompts.gsm import standard_prompt, cot_prompt, vote_prompt
import re


class GSMTask(Task):
    def __init__(self, num=100):
        self.dataset = load_dataset("openai/gsm8k", "main")

        og_len = len(self.dataset["test"])
        select_idxs = list(range(0, og_len, og_len // num))
        self.problems = [self.dataset["test"][i]["question"] for i in select_idxs]
        self.answers = [
            self.dataset["test"][i]["answer"].split("\n")[-1][len("#### ") :]
            for i in select_idxs
        ]
        self.steps = 2
        self.stops = ["\nAnswer:\n", None]

    def __len__(self):
        return len(self.problems)

    def get_input(self, idx: int):
        return self.problems[idx]

    def test_output(self, idx: int, output: str):
        try:
            answer_idx = output.lower().index("the answer is ")
        except ValueError:
            return {"r": 0}
        answer = output[answer_idx + len("the answer is ") :]
        try:
            answer = re.findall(r"\d+", answer)[0]
        except IndexError:
            return {"r": 0}
        return {"r": int(answer == self.answers[idx])}

    @staticmethod
    def standard_prompt_wrap(x: str, y: str = "") -> str:
        return standard_prompt.format(input=x) + y

    @staticmethod
    def cot_prompt_wrap(x: str, y: str = "") -> str:
        return cot_prompt.format(input=x) + y

    @staticmethod
    def vote_prompt_wrap(x: str, ys: list) -> str:
        # TODO this prompt doesn't include the instruction
        prompt = vote_prompt
        prompt += f"Problem:\n{x}\n"
        for i, y in enumerate(ys, 1):
            prompt += f"Choice {i}:\n{y}\n"
        return prompt

    @staticmethod
    def vote_outputs_unwrap(vote_outputs: list, n_candidates: int) -> list:
        vote_results = [0] * n_candidates
        for vote_output in vote_outputs:
            pattern = r".*best choice is .*(\d+).*"
            match = re.match(pattern, vote_output, re.DOTALL)
            if match:
                vote = int(match.groups()[0]) - 1
                if vote in range(n_candidates):
                    vote_results[vote] += 1
            else:
                print(f"vote no match: {[vote_output]}")
        return vote_results


if __name__ == "__main__":
    task = GSMTask(100)
    # print(task.answers)
