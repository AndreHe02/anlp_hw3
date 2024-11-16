gsm8k_format = '"the answer is n" where n is a number'

standard_prompt = 'Answer the following question with "the answer is n" where n is a number. Only output "the answer is n": {input}.'

cot_prompt = """Answer the following question: {input}

Make a strategy then write. Your output should be of the following format:

Strategy:
Your strategy about how to answer the question.

Answer:
Your answer to the question. It should end with "the answer is n" where n is a number."""

vote_prompt = """Given a problem and several strategies, decide which choice is most promising. Analyze each choice in detail, then conclude in the last line "The best choice is {s}", where s is the integer id of the choice.
"""
