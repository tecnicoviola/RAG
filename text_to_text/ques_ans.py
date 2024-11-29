from transformers import pipeline

# Load the pre-trained model for question answering
qa_pipeline = pipeline("question-answering")

# Define the context and question
context = """
Hugging Face is a company that provides open-source libraries for Natural Language Processing (NLP). 
Their most popular library is called Transformers, which has pre-trained models for tasks like 
text classification, named entity recognition, question answering, and more. Hugging Face also 
provides an easy-to-use interface for working with these models.
"""
question = "What is the name of Hugging Face's popular library?"

# Get the answer from the model
answer = qa_pipeline(question=question, context=context)

# Print the question and the model's answer
print(f"Question: {question}")
print(f"Answer: {answer['answer']}")
