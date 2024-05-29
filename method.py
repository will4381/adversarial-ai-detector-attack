import requests
import torch
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from torch.nn.functional import softmax
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('gptzero_api_key')

# GPT Zero Detector - Add your API key to .env or swap to another detector
def ai_detector(generated_text):
    gptzero_api_key = api_key
    url = "https://api.gptzero.me/v2/predict/text"
    payload = {"document": generated_text, "version": "2024-04-04"}
    headers = {
        "x-api-key": gptzero_api_key,
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        data = response.json()
        if data['documents']:
            class_probabilities = data['documents'][0]['class_probabilities']
            return class_probabilities
        else:
            return {'ai': 0, 'human': 0, 'mixed': 0}
    else:
        return {'ai': 0, 'human': 0, 'mixed': 0}

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
detector_model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
mlm_model = pipeline("fill-mask", model="bert-base-uncased")

def compute_gradients(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    embeddings = model.bert.embeddings.word_embeddings(inputs['input_ids'])
    embeddings.retain_grad()
    outputs = model(inputs_embeds=embeddings, attention_mask=inputs['attention_mask'])
    loss = torch.nn.CrossEntropyLoss()(outputs.logits, torch.tensor([1]).to(outputs.logits.device))
    loss.backward()
    gradients = embeddings.grad
    return gradients

def calculate_word_importance(gradients, tokenizer, input_ids):
    token_importance = gradients.abs().sum(dim=-1).squeeze().cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().cpu().numpy())
    return dict(zip(tokens, token_importance))

def generate_adversarial_example(text, importance_scores, mlm_model, num_replacements=5):
    tokens = tokenizer.tokenize(text)
    sorted_importance = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
    top_tokens = [word for word, _ in sorted_importance[:num_replacements]]
    
    for token in top_tokens:
        if token in tokens:
            mask_index = tokens.index(token)
            tokens[mask_index] = tokenizer.mask_token
            masked_text = tokenizer.convert_tokens_to_string(tokens)
            suggestions = mlm_model(masked_text)
            best_replacement = suggestions[0]['token_str']
            tokens[mask_index] = best_replacement
    
    return tokenizer.convert_tokens_to_string(tokens)

# Text you wish to test goes here
text = """Biodiversity is vital for ecosystem health and stability, providing resilience against environmental changes and supporting a wide range of ecosystem services. It ensures food security, medicinal resources, and maintains natural cycles, contributing to overall planetary well-being."""

gradients = compute_gradients(text, detector_model, tokenizer)
input_ids = tokenizer(text, return_tensors='pt')['input_ids']
importance_scores = calculate_word_importance(gradients, tokenizer, input_ids)

adversarial_example = generate_adversarial_example(text, importance_scores, mlm_model)

original_score = ai_detector(generated_text=text)
adversarial_score = ai_detector(generated_text=adversarial_example)

print("Original Text:", text) # The original text you used
print("Original Score:", original_score) # GPT Zero score on the original text
print("Adversarial Text:", adversarial_example) # Adversarially generated example using fill mask
print("Adversarial Score:", adversarial_score) # GPT Zero score on adversarial text
