import re


# Get a normalized classification string from the model's raw response
def extract_confidence(text):
    if re.search(r"low", text, re.IGNORECASE):
        return "low"
    elif re.search(r"medium", text, re.IGNORECASE):
        return "medium"
    elif re.search(r"very high", text, re.IGNORECASE):
        return "very high"
    elif re.search(r"high", text, re.IGNORECASE):
        return "high"
    elif re.search(r"i don't know", text, re.IGNORECASE):
        return "idk"
    else:
        return "N/A"
    

def get_zero_shot_prompt(statement): 
    return f"""
You are a helpful and knowledgeable climate science and policy assistant trained to assess human expert confidence in statements about climate change.

You will be presented with a sentence about climate science, the impacts of climate change, or mitigation of climate change, retrieved or paraphrased from the 6th IPCC assessment report. 

This statement has been labeled as low, medium, high, or very high confidence by a consensus of climate experts, based on the type, quality, quantity, and consistency of scientific evidence available.

Respond *only* with one of the following words: 'low', 'medium', 'high', 'very high'.

Statement: {statement}
Confidence: """