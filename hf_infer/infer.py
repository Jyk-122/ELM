from transformers import AutoTokenizer
from modeling_qwen.modeling_qwen2_elm import Qwen2ELMForCausalLM

# model_name = "/data1/jiangyikun/models/Qwen2.5-1.5B-Instruct"
model_name = "./hf_infer/Qwen2.5-1.5B-Instruct-ELM"

model = Qwen2ELMForCausalLM.from_pretrained(
    model_name,
    torch_dtype="float",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Please solve the following problem step by step. When you reach the answer, please output the answer after 'The answer is: ' at the end of the response.\n\nJanet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=128,
    eos_token_id=151645, # <|im_end|>
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(response)