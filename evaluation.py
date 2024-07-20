import datasets
import tempfile
import logging
import random
import config
import os
import yaml
import logging
import difflib
import pandas as pd

import transformers
import datasets
import torch

from tqdm import tqdm

from common import laminiDocsFilename
from utilities import *
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)
global_config = None

fine_tuning_dataset_loaded = datasets.load_dataset("json", data_files=laminiDocsFilename, split="train")