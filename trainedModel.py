import lamini
from llama import BasicModelRunner

lamini.api_key = "ceefc6b819fc7fb17e02ea7332283f301e5d6b02a38bd67e1ac360cbf8615dc1"

model = BasicModelRunner("meta-llama/Llama-2-7b-hf")
#model = BasicModelRunner("meta-llama/Llama-2-7b-chat-hf")
non_finetuned_output = model(
    "[INST]" + "Assess the agent's response and suggest a better one if needed." + "[INST]"
                                                                                   "customer query: I received a damaged product." +
    "agent response : That's your fault for not checking the package before accepting it." \
    "agent response assessment: " \
    "suggested response: ")
print(non_finetuned_output)