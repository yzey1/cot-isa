from prompt_template import SentimentAnalysisTemplates
from openai import OpenAI
import ollama
import yaml

# config

config = yaml.load(open("config.yaml", 'r'), Loader=yaml.FullLoader)
api_key_dict = config['api_key_dict']
base_url_dict = config['base_url_dict']
label_list = config['label_list']
templates = SentimentAnalysisTemplates()


# get llm response

def get_llm_response(conversation, prompt_text, model_name='llama3.1'):
    
    conversation.append(
        {'role': 'user', "content": prompt_text}
    )
    
    if model_name.startswith("llama"):
        response = ollama.chat(model=model_name, messages=conversation)
        content = response['message']['content']
        
    else:
        client = OpenAI(
            api_key=api_key_dict[model_name], 
            base_url=base_url_dict[model_name], 
        )
        completion = client.chat.completions.create(
            model = model_name,
            messages=conversation,
        )
        content = completion.choices[0].message.content
        
    conversation.append(
        {"role": "assistant", "content": content}
    )
    
    result = content.replace('\n', ' ').strip()
    return conversation, result

# direct inference

def direct_inference(text, target, model_name):
    
    conversation = [{'role': 'system', "content": templates.system_prompt}]
    
    context_step1, prompt_step1 = templates.prompt_direct_inferring(text, target)
    conversation, output_lb = get_llm_response(conversation, prompt_step1, model_name)
    
    output_lb = output_lb.lower().strip()
    output = 2
    for k, lb in enumerate(label_list):
        if lb in output_lb: output = k; break
    
    return conversation, output


# cot inference

def cot_inference(text, target, model_name):
    
    conversation = [{'role': 'system', "content": templates.system_prompt}]
    
    # step 1: aspect inferring
    context_step1, prompt_step1 = templates.prompt_for_aspect(text, target)
    conversation, aspect_expr = get_llm_response(conversation, prompt_step1, model_name)

    # step 2: opinion inferring
    context_step2, prompt_step2 = templates.prompt_for_opinion(context_step1, target, aspect_expr)
    conversation, opinion_expr = get_llm_response(conversation, prompt_step2, model_name)

    # step 3: polarity inferring
    context_step3, prompt_step3 = templates.prompt_for_polarity(context_step2, target, opinion_expr)
    conversation, output_lb = get_llm_response(conversation, prompt_step3, model_name)
    
    # get the output label
    output_lb = output_lb.lower().strip()
    output = 2
    for k, lb in enumerate(label_list):
        if lb in output_lb: output = k; break
    
    return conversation, output
    