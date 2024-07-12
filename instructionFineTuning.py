import json

from lamini.api import lamini
from llama import BasicModelRunner
from common import processedDataFileName

model = BasicModelRunner("EleutherAI/pythia-70m")

with open(processedDataFileName, 'r', encoding='utf8') as f:
    data = json.load(f)
    model.load_data(data)
    model.train()

