from pprint import pprint

from transformers import pipeline

'''
generator = pipeline("text-generation", model="distilgpt2")


res = generator(
    "In this course, we will teach you how to",
    max_length=300,
    num_return_sequences=3,
    truncation=True
)
'''

classifier = pipeline("zero-shot-classification")
res = classifier("This is a course about python list comprehension",
                 candidate_labels=["education", "politics", "business"])

pprint(res)