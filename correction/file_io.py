import os
import json
from openai import OpenAI, AzureOpenAI
from tqdm import tqdm
import numpy as np
import re
import time
import datetime
import tiktoken
import os

os.environ['http_proxy'] = 'http://agent.baidu.com:8891'
os.environ['https_proxy'] = 'http://agent.baidu.com:8891'


def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def run_gpt(messages, model="gpt-4o-mini", **kwargs):
    client = OpenAI(api_key='')
    completion = client.chat.completions.create(model=model, messages=messages, **kwargs)
    return [completion.choices[i].message.content for i in range(len(completion.choices))]


def run_azure(messages, model="gpt-4o-mini", **kwargs):
    client = AzureOpenAI(
        azure_endpoint="https://weiwes.openai.azure.com/",
        api_key="",
        api_version="2024-02-01"
    )
    completion = client.chat.completions.create(model=model, messages=messages, **kwargs)
    return [completion.choices[i].message.content for i in range(len(completion.choices))]


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


def all_file(dirname):
    fl = []
    for root, dirs, files in os.walk(dirname):
        for item in files:
            path = os.path.join(root, item)
            fl.append(path)
    return fl


def read_file(file):
    out = []
    with open(file) as f:
        for line in f:
            out.append(line)
    return out


def read_json(file):
    if not os.path.exists(file):
        return {}
    return json.load(open(file))


def read_jsonl(file):
    out = []
    for line in read_file(file):
        out.append(json.loads(line))
    return out


def create_dir(file_path):
    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def write_json(obj, file):
    create_dir(file)
    with open(file, 'w') as f:
        json.dump(obj, f)


def write_jsonl(obj, file):
    create_dir(file)
    with open(file, 'w') as f:
        for line in obj:
            f.write(json.dumps(line) + '\n')


def add_jsonl(obj, file):
    create_dir(file)
    with open(file, 'a') as f:
        for line in obj:
            f.write(json.dumps(line) + '\n')


def math_normalizer(text: str) -> str:  # noqa C901
    """Source: https://github.com/hendrycks/math"""

    def _remove_boxed(text) -> str:
        """
        Extract the text within a \\boxed{...} environment.
        Example:
        >>> _remove_boxed(\\boxed{\\frac{2}{3}})
        \\frac{2}{3}
        """
        if text is None:
            return ""
        if "\\boxed " in text:
            left = "\\boxed "
            assert text[: len(left)] == left
            return text[len(left):]

        left = "\\boxed{"

        assert text[: len(left)] == left
        assert text[-1] == "}"

        return text[len(left): -1]

    def _last_boxed_only_string(text):
        """Extract the last \\boxed{...} or \\fbox{...} element from a string."""
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
            retval = None
        else:
            retval = text[idx: right_brace_idx + 1]

        return retval

    def _fix_fracs(text: str) -> str:
        """
        Fix the formatting of fractions in the given text.
        Copied from: https://github.com/hendrycks/math/blob/357963a7f5501a6c1708cf3f3fb0cdf525642761/modeling/math_equivalence.py#L1

        Args:
            text (str): The input text.

        Returns:
            str: The text with properly formatted fractions.

        Examples:
            >>> _fix_fracs("\\frac12")
            "\\frac{1}{2}"
            >>> _fix_fracs("\\frac{3}{4}")
            "\\frac{3}{4}"
            >>> _fix_fracs("\\frac1{2}")
            "\\frac{1}{2}"
        """
        substrs = text.split("\\frac")
        new_str = substrs[0]
        if len(substrs) > 1:
            substrs = substrs[1:]
            for substr in substrs:
                new_str += "\\frac"
                if substr[0] == "{":
                    new_str += substr
                else:
                    try:
                        assert len(substr) >= 2
                    except AssertionError:
                        return text
                    a = substr[0]
                    b = substr[1]
                    if b != "{":
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}{" + b + "}" + post_substr
                        else:
                            new_str += "{" + a + "}{" + b + "}"
                    else:
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}" + b + post_substr
                        else:
                            new_str += "{" + a + "}" + b
        text = new_str
        return text

    def _fix_a_slash_b(text: str) -> str:
        """Source: https://github.com/hendrycks/math
        Reformat fractions formatted as a/b to \\frac{a}{b}.
        Example:
        >>> _fix_a_slash_b("2/3")
        \frac{2}{3}
        """
        if len(text.split("/")) != 2:
            return text
        a_str = text.split("/")[0]
        b_str = text.split("/")[1]
        try:
            a = int(a_str)
            b = int(b_str)
            assert text == "{}/{}".format(a, b)
            new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
            return new_string
        except Exception:
            return text

    def _remove_right_units(text: str) -> str:
        """
        Removes unit descriptions from LaTeX-formatted text, where units are indicated by "\\text{ }".
        This function splits the text at each "\\text{ " and returns the part before the first occurrence,
        effectively discarding any units and additional text following this pattern. This function also
        trims any trailing whitespace left after removing units.

        Args:
            text (str): The input string potentially containing LaTeX-style unit descriptions.

        Returns:
            str: The text with unit descriptions removed.

        Examples:
            - Input: '50.5 \\text{ kg}'
            Output: '50.5'

            - Input: 'The mass is 20 grams'
            Output: 'The mass is 20 grams'

            - Input: 'The object weighs 30.2 \\text{ lbs} and is 15 \\text{ inches} long'
            Output: 'The object weighs 30.2'

            - Input: '\\text{ unit without preceding text}'
            Output: ''

        Note:
            This function assumes that "\\text{ " is only used to denote units. It will remove all text
            following the first occurrence of "\\text{ ", including any further text and units that might
            appear in complex sentences.
        """
        # Check for "\\text{ " and split the text at each occurrence
        if "\\text{ " in text:
            splits = text.split("\\text{ ")
            # Return only the first part which is assumed to contain the main content without units
            return splits[0].rstrip()
        else:
            return text

    def _fix_sqrt(text: str) -> str:
        """Source: https://github.com/hendrycks/math
        Reformat square roots.
        Example:
        >>> _fix_sqrt("\\sqrt3")
        \\sqrt{3}
        """
        if "\\sqrt" not in text:
            return text
        splits = text.split("\\sqrt")
        new_string = splits[0]
        for split in splits[1:]:
            if split[0] != "{":
                a = split[0]
                new_substr = "\\sqrt{" + a + "}" + split[1:]
            else:
                new_substr = "\\sqrt" + split
            new_string += new_substr
        return new_string

    text = _remove_boxed(_last_boxed_only_string(text))

    to_replace_1 = [
        ("\n", ""),  # linebreaks
        ("\\!", ""),  # remove inverse spaces
        ("\\\\", "\\"),  # replace \\ with \
        ("tfrac", "frac"),  # replace tfrac and dfrac with frac
        ("dfrac", "frac"),
        ("\\left", ""),  # remove \left and \right
        ("\\right", ""),
        ("^{\\circ}", ""),  # Remove circ (degrees)
        ("^\\circ", ""),
        ("\\$", ""),  # remove dollar signs
    ]

    for input_str, output_str in to_replace_1:
        text = text.replace(input_str, output_str)

    # remove units (on the right)
    text = _remove_right_units(text)

    to_replace_2 = [
        ("\\%", ""),  # remove percentage
        (r"\%", ""),
        (
            " .",
            " 0.",
        ),
        # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the text
        ("{.", "{0."),
    ]
    for input_str, output_str in to_replace_2:
        text = text.replace(input_str, output_str)

    # if empty, return empty text
    if len(text) == 0:
        return text
    if text[0] == ".":
        text = "0" + text

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(text.split("=")) == 2:
        if len(text.split("=")[0]) <= 2:
            text = text.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    text = _fix_sqrt(text)

    # remove spaces
    text = text.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    text = _fix_fracs(text)

    # manually change 0.5 --> \frac{1}{2}
    # if text == "0.5":
    #     text = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    text = _fix_a_slash_b(text)

    return text

