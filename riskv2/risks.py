"""
Functions to calculate risks given the classification of the passages in a document. Works at document level, so results contain the extracted passages containing the keywords and their classifications either from BERT or GPT
"""

from sentence_transformers import SentenceTransformer, util

# Load pre-trained model
# model = SentenceTransformer('all-MiniLM-L6-v2')
model = SentenceTransformer("all-roberta-large-v1")

NEGATIVE_PASSAGE = """
We are cautiously exploring the use of AI due to potential ethical concerns, as the implementation of AI poses significant regulatory and compliance challenges, potentially disrupting traditional processes, leading to operational risks, increasing exposure to cybersecurity vulnerabilities, involving high upfront costs and uncertain return on investment, raising concerns about job displacement and societal backlash, amplifying bias in AI algorithms, causing reputational risks if errors occur, and creating challenges for staying competitive due to the lack of clear legal frameworks around AI usage.
"""

POSITIVE_PASSAGE = """
AI represents a transformative opportunity to enhance operational efficiency by driving innovation across all business units, unlocking new business opportunities, reducing errors, improving customer satisfaction, aligning with our long-term growth strategy, enabling enhanced decision-making capabilities, helping us stay ahead in a competitive market, supporting our commitment to sustainability and societal impact, and ensuring ethical development to align with our values and goals, demonstrating that the potential of AI far outweighs the risks when implemented responsibly.
"""

neg_emb = model.encode(NEGATIVE_PASSAGE, convert_to_tensor=True)
pos_emb = model.encode(POSITIVE_PASSAGE, convert_to_tensor=True)

def naive_risk(results):

    predictions = []
    for r in results:
        prediction = r[0]
        # Failsafe for GPT (check last comments, end of script)
        if "NEGATIVE" in prediction and "POSITIVE" not in prediction:
            prediction = "NEGATIVE"
        elif "POSITIVE" in prediction and "NEGATIVE" not in prediction:
            prediction = "POSITIVE"

        if prediction == "NEGATIVE":
            predictions.append(0)
        elif prediction == "POSITIVE":
            predictions.append(1)

    if len(predictions) == 0:
        return None

    cz = predictions.count(0)
    risk = cz/len(predictions)
    
    return risk


def np_risk(results):

    for r in results:
        passage = r[1]
        passage_emb = model.encode(passage, convert_to_tensor=True)
    
        p_similarity = util.pytorch_cos_sim(passage_emb, pos_emb)
        p_sim = p_similarity.item()

        n_similarity = util.pytorch_cos_sim(passage_emb, neg_emb)
        n_sim = n_similarity.item()

        print("*"*100)
        print(r[0])
        print("*"*100)

        print(passage, p_sim, n_sim)
        print("*"*100)

    return 1