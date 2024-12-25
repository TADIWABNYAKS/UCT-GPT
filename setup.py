'''
Script to download gptNeo-3.7B and run a simple test prompt to confirm it has been setup 
'''
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_name = "EleutherAI/gpt-neo-2.7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", low_cpu_mem_usage=True)

#INPUT
prompt = "Are you working?"
inputs = tokenizer(prompt, return_tensors="pt")


#MODEL OUTPUT
outputs = model.generate(**inputs, max_length=50)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
