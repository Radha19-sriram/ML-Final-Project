# Question Answering System with Streamlit

This repository contains a simple question answering system built using Streamlit, Hugging Face's Transformers library, and PyTorch. The system takes a question and context as input and generates an answer using a pre-trained DistilBERT model fine-tuned on the SQuAD dataset. Additionally, it computes F1 and ROUGE scores to evaluate the quality of the generated answer.

## Features

- **Question Answering**: Generates answers based on the provided question and context.
- **F1 Score Calculation**: Computes the F1 score to evaluate the accuracy of the generated answer.
- **ROUGE Score Calculation**: Computes ROUGE-1, ROUGE-2, and ROUGE-L scores to evaluate the quality of the generated answer.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Radha19-sriram/question-answering-system.git
cd question-answering-system

2. Install the required dependencies
pip install -r requirements.txt

3. Run the Streamlit app:
streamlit run LLM_project.py

# How It Works
1.**Loading the Model and Tokenizer:** The app loads a pre-trained DistilBERT model and its tokenizer from Hugging Face's model hub.
2.**Question Answering:** The app tokenizes the input question and context, then uses the model to predict the start and end positions of the answer in the context.
3.**F1 Score Calculation:** The app computes the F1 score based on the overlap between the generated answer and the true answer provided by the user.
4.**ROUGE Score Calculation:** The app computes ROUGE-1, ROUGE-2, and ROUGE-L scores to measure the quality of the generated answer.

To use the app, follow these steps:

1. Enter your question in the "Enter your question" field.
2. Enter the context in the "Enter the context" field.
3. Optionally, enter the true answer in the "Enter the true answer" field for F1 and ROUGE score calculations.
4. Click the "Get Answer" button to generate the answer and view the scores.
Requirements
Python 3.7 or higher
streamlit
transformers
torch
rouge_score

Contributing
Contributions are welcome! Please open an issue or submit a pull request if you have any suggestions or improvements.

Acknowledgements
Hugging Face for providing the Transformers library and pre-trained models.
Streamlit for providing an easy-to-use framework for building web apps.
PyTorch for the deep learning framework.

Contact
If you have any questions or feedback, feel free to reach out.

Email: radharangarajan1988@gmail.com
