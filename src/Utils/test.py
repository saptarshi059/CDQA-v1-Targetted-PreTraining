import torch, gc
from transformers import AutoTokenizer, OPTForCausalLM, set_seed
import re

tokenizer = AutoTokenizer.from_pretrained("facebook/galactica-1.3b")
tokenizer.pad_token_id = 1
tokenizer.padding_side = 'right'
tokenizer.model_max_length = 2048

model = OPTForCausalLM.from_pretrained("facebook/galactica-1.3b", device_map="auto", torch_dtype=torch.float16)

final_string = ''
prev_new_text = ''
N = 80

init_string = '(coronavirus) a possible cause of myasthenia gravis'

while len(tokenizer(final_string)['input_ids']) < 5000:
  print(len(tokenizer(final_string)['input_ids']))
  if final_string == '':
    input_text = f'Title: {init_string}'
    input_ids = tokenizer(input_text, return_tensors="pt", padding='max_length').input_ids.to("cuda")
    set_seed(42)
    outputs = model.generate(input_ids,
                        max_new_tokens=2000,
                        do_sample=True,
                        temperature=0.9,
                        top_k=25,
                        top_p=0.9,
                        no_repeat_ngram_size=10,
                        early_stopping=True,
                        renormalize_logits=True, 
                        use_cache=True)

    final_string = tokenizer.decode(outputs[0]).lstrip('<pad>')
    gc.collect()
    torch.cuda.empty_cache()

  else:
    final_string_tokenized = tokenizer(final_string)
    input_text = tokenizer.decode(final_string_tokenized['input_ids'][-N:])
    input_ids = tokenizer(input_text, return_tensors="pt", padding='max_length').input_ids.to("cuda")
    set_seed(42)
    outputs = model.generate(input_ids,
                        max_new_tokens=2000,
                        do_sample=True,
                        temperature=0.9,
                        top_k=25,
                        top_p=0.9,
                        no_repeat_ngram_size=10,
                        early_stopping=True,
                        renormalize_logits=True, 
                        use_cache=True)

    new_text = tokenizer.decode(outputs[0]).lstrip('<pad>')
    
    if prev_new_text == new_text:
      break
    else:
      prev_new_text = new_text
    
    final_string = final_string + '' + re.sub(re.escape(input_text), '', new_text)
    gc.collect()
    torch.cuda.empty_cache()

with open('output.txt', 'w') as f:
	f.write(final_string)