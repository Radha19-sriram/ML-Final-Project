import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import collections
from rouge_score import rouge_scorer

# Load pre-trained model and tokenizer
model_name = "distilbert-base-cased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Define function for question answering
def answer_question(question, context):
    inputs = tokenizer(question, context, return_tensors="pt")
    outputs = model(**inputs)
    
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits
    
    start_index = torch.argmax(start_logits)
    end_index = torch.argmax(end_logits) + 1

    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_index:end_index]))
    return answer

# Compute F1 score manually
def compute_f1_score(pred_text, true_text):
    pred_tokens = pred_text.lower().split()
    true_tokens = true_text.lower().split()
    
    # Compute intersection
    common = collections.Counter(true_tokens) & collections.Counter(pred_tokens)
    num_common = sum(common.values())
    
    # Compute precision and recall
    precision = num_common / len(pred_tokens) if len(pred_tokens) > 0 else 0
    recall = num_common / len(true_tokens) if len(true_tokens) > 0 else 0
    
    # Compute F1 score
    if precision + recall != 0:
        f1_score = (2 * precision * recall) / (precision + recall)
    else:
        f1_score = 0.0
    
    return f1_score

# Compute ROUGE scores
def compute_rouge_scores(pred_text, true_text):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(pred_text, true_text)
    return scores

# Streamlit app
def main():
    st.title("Question Answering System")

    # Input fields for question and context
    question = st.text_input("Enter your question:")
    context = st.text_area("Enter the context:")
    true_answer = st.text_input("Enter the true answer (for F1 score calculation):")

    # Button to perform question answering
    if st.button("Get Answer"):
        if not question.strip() or not context.strip():
            st.error("Please provide both question and context.")
        else:
            answer = answer_question(question, context)
            st.write("Generated Answer:", answer)

            if true_answer.strip():
                # Compute F1 score
                f1_score = compute_f1_score(answer, true_answer)
                st.write("F1 Score:", f1_score)

                # Compute ROUGE scores
                rouge_scores = compute_rouge_scores(answer, true_answer)
                for metric, value in rouge_scores.items():
                    st.write(f"{metric.upper()} Score:", value)
            else:
                st.warning("Please provide the true answer for F1 score calculation.")

if __name__ == "__main__":
    main()
