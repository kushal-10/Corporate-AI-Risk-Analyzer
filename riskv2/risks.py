"""
Functions to calculate risks given the classification of the passages in a document. Works at document level, so results contain the extracted passages containing the keywords and their classifications either from BERT or GPT
"""

from sentence_transformers import SentenceTransformer, util

# Load pre-trained model
# model = SentenceTransformer('all-MiniLM-L6-v2')
model = SentenceTransformer("all-roberta-large-v1")

NEGATIVE_PASSAGE = """
Rapid advancements in AI introduce significant competition across industries and geographies, exposing us to increased risks from cybersecurity vulnerabilities, regulatory uncertainty, and operational disruptions. Intense competition, fueled by new technologies like AI and machine learning, pressures pricing strategies, product success, and market share, requiring higher investment and reducing profits. Expansion into AI-related products and services exposes us to limited expertise, potential quality issues, and challenges in achieving expected benefits, risking financial losses. Unforeseen geopolitical conditions and sustainability efforts may exacerbate these risks, amplifying the adverse effects on our operations and profitability in a rapidly evolving AI landscape.
"""

POSITIVE_PASSAGE = """
Through initiatives like introducing computer science classes in underserved schools, awarding scholarships to students from low-income backgrounds, and providing guaranteed internships, we ensure AI's positive societal impact. By investing efficiently in technology like AWS, wireless connectivity, and practical applications of AI, we enhance operational efficiency, customer experience, and innovation across industries. With programs like Career Choice and upskilling efforts in machine learning and cloud computing, we empower employees to evolve with technology. By aligning compensation strategies with shareholders' interests, leveraging innovative software and electronic devices, and maintaining a focus on sustainability, we demonstrate how AI drives growth, opportunity, and societal benefit while aligning with long-term goals.
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

    sub_risks = []
    for r in results:
        passage = r[1]
        prediction = r[0]
        C = 0
        # Failsafe for GPT (check last comments, end of script)
        if "NEGATIVE" in prediction and "POSITIVE" not in prediction:
            prediction = "NEGATIVE"
        elif "POSITIVE" in prediction and "NEGATIVE" not in prediction:
            prediction = "POSITIVE"

        if prediction == "NEGATIVE":
            C = 1
        elif prediction == "POSITIVE":
            C = 0
        else:
            continue

        passage_emb = model.encode(passage, convert_to_tensor=True)
    
        p_similarity = util.pytorch_cos_sim(passage_emb, pos_emb)
        p_sim = p_similarity.item()
        n_similarity = util.pytorch_cos_sim(passage_emb, neg_emb)
        n_sim = n_similarity.item()

        subrisk = (C*(n_sim/p_sim)) - ((1-C)*(p_sim/n_sim))

        sub_risks.append(subrisk)

    if not sub_risks:
        return None
    
    else:
        return sum(sub_risks)/len(sub_risks)
