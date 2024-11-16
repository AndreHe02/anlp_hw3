import os
import openai
import backoff

completion_tokens = prompt_tokens = 0

# api_choice = "openai"  # or "litellm"
api_choice = "sglang"

if api_choice == "openai":
    api_key = os.getenv("OPENAI_API_KEY", "")
elif api_choice == "litellm":
    api_key = os.getenv("LITELLM_API_KEY")
elif api_choice == "sglang":
    api_key = ""

if api_key != "":
    openai.api_key = api_key
else:
    print("Warning: OPENAI_API_KEY is not set")

# openai.base_url = "https://cmu.litellm.ai"

# api_base = os.getenv("OPENAI_API_BASE", "")
# if api_base != "":
#     print("Warning: OPENAI_API_BASE is set to {}".format(api_base))
#     openai.api_base = api_base

# client = openai.OpenAI(api_key=api_key, base_url="https://cmu.litellm.ai")
if api_choice == "openai":
    client = openai.OpenAI(api_key=api_key)
elif api_choice == "litellm":
    client = openai.OpenAI(api_key=api_key, base_url="https://cmu.litellm.ai")
elif api_choice == "sglang":
    # client = openai.OpenAI(api_key=api_key)
    client = openai.OpenAI(api_key="None", base_url="http://127.0.0.1:30000/v1")


# @backoff.on_exception(backoff.expo, openai.APIError)
def completions_with_backoff(**kwargs):
    # return openai.ChatCompletion.create(**kwargs)
    return client.chat.completions.create(**kwargs)


def gpt(
    prompt, model="gpt-4", temperature=0.7, max_tokens=1000, n=1, stop=None
) -> list:
    messages = [{"role": "user", "content": prompt}]
    return chatgpt(
        messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        n=n,
        stop=stop,
    )


def chatgpt(
    messages, model="gpt-4", temperature=0.7, max_tokens=1000, n=1, stop=None
) -> list:
    global completion_tokens, prompt_tokens
    outputs = []
    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        res = completions_with_backoff(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            n=cnt,
            stop=stop,
        )
        # outputs.extend([choice["message"]["content"] for choice in res["choices"]])
        # completion_tokens += res["usage"]["completion_tokens"]
        # prompt_tokens += res["usage"]["prompt_tokens"]

        outputs.extend([choice.message.content for choice in res.choices])
        completion_tokens += res.usage.completion_tokens
        prompt_tokens += res.usage.prompt_tokens
        # log completion tokens
    return outputs


def gpt_usage(backend="gpt-4"):
    global completion_tokens, prompt_tokens
    if backend == "gpt-4":
        cost = completion_tokens / 1000 * 0.06 + prompt_tokens / 1000 * 0.03
    elif backend == "gpt-3.5-turbo":
        cost = completion_tokens / 1000 * 0.002 + prompt_tokens / 1000 * 0.0015
    elif backend == "gpt-4-turbo":
        cost = completion_tokens / 1000 * 0.03 + prompt_tokens / 1000 * 0.01
    elif backend == "davinci-002":
        cost = completion_tokens / 1000 * 0.002 + prompt_tokens / 1000 * 0.002
    else:
        cost = 0
    return {
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "cost": cost,
    }
