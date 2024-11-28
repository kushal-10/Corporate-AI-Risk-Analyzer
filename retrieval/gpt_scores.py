from openai import OpenAI
client = OpenAI()

BASE_PROMPT = """
Classify a given SENTENCE about Artificial Intelligence (AI) as either POSITIVE or NEGATIVE. POSITIVE reflects AI as a reward or opportunity, while NEGATIVE includes risks, challenges, or neutral views without a clear positive stance. SENTENCE : \n 
"""
def classify(input_prompt: str):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": BASE_PROMPT},
            {
                "role": "user",
                "content": input_prompt
            }
        ]
    )

    return completion.choices[0].message.content

if __name__=='__main__':
    input_prompt = "Artificial Intelligence"
    label = classify(input_prompt) 
    print(label, type(label)) # NEGATIVE, STR