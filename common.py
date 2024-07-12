import lamini
from llama import BasicModelRunner
from transformers import AutoTokenizer

lamini.api_key = "ceefc6b819fc7fb17e02ea7332283f301e5d6b02a38bd67e1ac360cbf8615dc1"
laminiDocsFilename = "lamini_docs.jsonl"
processedDataFileName = "lamini_docs_processed.jsonl"
taylor_swift_dataset = "lamini/taylor_swift"
bts_dataset = "lamini/bts"
open_llms = "lamini/open_llms"
model_pythia_70m_name = "EleutherAI/pythia-70m"

tokenizer = AutoTokenizer.from_pretrained(model_pythia_70m_name)
tokenizer.pad_token = tokenizer.eos_token


def inference(text, model, max_input_tokens=100, max_output_tokens=100):
    # Tokenize
    input_ids = tokenizer.encode(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens
    )

    # Generate
    device = model.device
    generated_tokens_with_prompt = model.generate(
        input_ids=input_ids.to(device),
        max_length=max_output_tokens
    )

    # Decode
    generated_text_with_prompt = tokenizer.batch_decode(generated_tokens_with_prompt, skip_special_tokens=True)

    # Strip the prompt
    generated_text_answer = generated_text_with_prompt[0][len(text):]

    return generated_text_answer

