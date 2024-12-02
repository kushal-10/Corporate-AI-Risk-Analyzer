import os
import json
from math import sqrt

def create_confusion_matrix(list1, list2):
    # Initialize counters
    pos_pos = 0  # Both lists say POSITIVE
    pos_neg = 0  # List1 says POSITIVE, List2 says NEGATIVE
    neg_pos = 0  # List1 says NEGATIVE, List2 says POSITIVE
    neg_neg = 0  # Both lists say NEGATIVE
    
    total = len(list1)
    
    for l1, l2 in zip(list1, list2):
        if l1 == "POSITIVE" and l2 == "POSITIVE":
            pos_pos += 1
        elif l1 == "POSITIVE" and l2 == "NEGATIVE":
            pos_neg += 1
        elif l1 == "NEGATIVE" and l2 == "POSITIVE":
            neg_pos += 1
        else:  # both negative
            neg_neg += 1
    
    return pos_pos, pos_neg, neg_pos, neg_neg

def calculate_metrics_from_confusion_matrix(pos_pos, pos_neg, neg_pos, neg_neg):
    total = pos_pos + pos_neg + neg_pos + neg_neg
    
    # Calculate observed agreement (percentage agreement)
    observed_agreement = (pos_pos + neg_neg) / total
    
    # Calculate expected agreement for Cohen's Kappa
    list1_pos = (pos_pos + pos_neg) / total
    list1_neg = (neg_pos + neg_neg) / total
    list2_pos = (pos_pos + neg_pos) / total
    list2_neg = (pos_neg + neg_neg) / total
    
    expected_agreement = (list1_pos * list2_pos) + (list1_neg * list2_neg)
    
    # Calculate Cohen's Kappa
    kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)
    
    # Convert confusion matrix to binary arrays for correlation
    list1 = [1] * (pos_pos + pos_neg) + [0] * (neg_pos + neg_neg)
    list2 = [1] * (pos_pos + neg_pos) + [0] * (pos_neg + neg_neg)
    
    # Calculate Pearson correlation
    mean1 = sum(list1) / len(list1)
    mean2 = sum(list2) / len(list2)
    
    numerator = sum((x - mean1) * (y - mean2) for x, y in zip(list1, list2))
    denominator = sqrt(
        sum((x - mean1) ** 2 for x in list1) *
        sum((y - mean2) ** 2 for y in list2)
    )
    pearson = numerator / denominator if denominator != 0 else 0

    print(f"""
        Confusion Matrix:
                        List2
        List1     POSITIVE  NEGATIVE
        POSITIVE    {pos_pos:^8d} {pos_neg:^8d}
        NEGATIVE    {neg_pos:^8d} {neg_neg:^8d}

        Total samples: {total}
        Agreement Percentage: {observed_agreement:.2%}
        Cohen's Kappa: {kappa:.3f}
        Pearson Correlation: {pearson:.3f}

        Interpretation:
        - Agreement < 0.40: Poor agreement
        - Agreement 0.40-0.75: Fair to good agreement
        - Agreement > 0.75: Excellent agreement

        Kappa Interpretation:
        - < 0: Poor agreement
        - 0.01-0.20: Slight agreement
        - 0.21-0.40: Fair agreement
        - 0.41-0.60: Moderate agreement
        - 0.61-0.80: Substantial agreement
        - 0.81-1.00: Almost perfect agreement
        
        Correlation Interpretation:
        - -1.0 to -0.5: Strong negative correlation
        - -0.5 to -0.1: Weak negative correlation
        -  -0.1 to 0.1: No correlation
        -   0.1 to 0.5: Weak positive correlation
        -   0.5 to 1.0: Strong positive correlation
    """)
    
    return observed_agreement, kappa, pearson



with open(os.path.join('retrieval', 'labels.json'), 'r') as f:
    label_data = json.load(f)

with open(os.path.join('retrieval', 'gpt_labels.json')) as f:
    gpt_data = json.load(f)

keys = list(label_data.keys())
check_set = set(['POSITIVE', 'NEGATIVE'])

gpt_results = []
db_results = []

total_predictions = 0
gpt_fails = 0
db_fails = 0

for k in keys:
    gpt_prediction = gpt_data[k]
    distillbert_prediction = label_data[k]

    try:
        assert (len(distillbert_prediction) == len(gpt_prediction))
    except:
        raise ValueError(f"Predictions for GPT and DistillBERT do not match for {k}")

    for i in range(len(gpt_prediction)):
        gpt = gpt_prediction[i][0]
        db = distillbert_prediction[i][0]

        # Failsafe for GPT (check last comments, end of script)
        if "NEGATIVE" in gpt and "POSITIVE" not in gpt:
            gpt = "NEGATIVE"
        elif "POSITIVE" in gpt and "NEGATIVE" not in gpt:
            gpt = "POSITIVE"

        if gpt not in check_set:
            print(f"Wrong prediction by GPT. Check {i+1} value for {k} - {gpt}")
            gpt_fails += 1

        if db not in check_set:
            print(f"Wrong prediction by BERT. Check {i+1} value for {k} - {db}")
            db_fails += 1

        gpt_results.append(gpt)
        db_results.append(db)
        total_predictions += 1


pos_pos, pos_neg, neg_pos, neg_neg = create_confusion_matrix(gpt_results, db_results)
agree, kappa, pearson = calculate_metrics_from_confusion_matrix(pos_pos, pos_neg, neg_pos, neg_neg)
print(f"GPT Failed : {gpt_fails}. = {gpt_fails*100/total_predictions}")
print(f"DB Failed : {db_fails}. = {db_fails*100/total_predictions}")


"""
Without Failsafe the model sometimes predicts something like -
- be enhanced with advanced artificial intelligence technologies to improve efficiency and customer experience. This strategic focus on AI integration exemplifies the commitment of Daimler Financial Services to leverage innovative solutions for better service delivery and operational effectiveness.POSITIVE 

Failed cases = 
GPT Failed : 776. = 5.865013982314262
DB Failed : 0. = 0.0

Failed Cases after Failsafe X|:
1) Wrong prediction by GPT. Check 6 value for USA/18.Johnson&Johnson_$395.70 B_Health Care/2021 - The sentence provided is predominantly neutral and focuses on financial reporting and adjustments related to company assets and research and development, without portraying a clear positive or negative stance specifically about Artificial Intelligence (AI). Thus, it doesn't fit well into either category of POSITIVE or NEGATIVE regarding Artificial Intelligence. 

2) Wrong prediction by GPT. Check 15 value for China/6.China Mobile_$206.74 B_Communication Services/2019 - The provided sentence does not clearly classify Artificial Intelligence (AI) in a positive or negative light. It discusses an individual's background and roles related to big data research without mentioning AI's implications, rewards, risks, or opportunities directly. Therefore, it reflects a neutral stance without a clear positive perspective.

Classification: NEUTRAL


"""