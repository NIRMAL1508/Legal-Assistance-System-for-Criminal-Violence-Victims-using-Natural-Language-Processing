# Legal-Assistance-System-for-Criminal-Violence-Victims-using-Natural-Language-Processing

## Overview
The "Legal Assistance System for Criminal Violence Victims" leverages Natural Language Processing (NLP) to provide legal guidance to individuals who are victims of criminal violence. Traditional legal processes can be confusing and inaccessible, often requiring victims to navigate complex legal terminology and procedures. This system is designed to simplify the process by allowing users to interact in natural language, making legal information more accessible and easier to understand.

## Functionality

### User Interaction: 
Victims can input their incidents in natural language, describing their experiences without needing to understand legal jargon.
### BERT-based NLP Processing: 
The system utilizes a fine-tuned BERT (Bidirectional Encoder Representations from Transformers) model to process user input, capturing contextual relationships and understanding the nuances of the language.
### Legal Guidance:
After processing, the system provides relevant legal advice, suggesting potential legal actions based on the description of the incident.
### Streamlit Interface: 
The system is deployed through a user-friendly Streamlit interface, enabling easy access and interaction.

## Key Features
### Tokenization & Semantic Analysis: 
Breaks down user input into tokens and analyzes the semantics to understand the context.
### Error Correction & POS Tagging: 
Corrects common errors and identifies parts of speech to enhance understanding.
### Contextual Word Embeddings: 
Generates word embeddings that capture the context, improving the accuracy of legal recommendations.
### Cosine Similarity Matching: 
Measures the similarity between user queries and pre-defined legal sections to provide the most relevant advice.
### L2 Normalization: 
Ensures uniformity in vector magnitude, improving the consistency of results.
### Top-10 Recommendation Accuracy: 
Utilizes a top-10 ranking system to provide the most relevant legal actions, improving decision-making for the user.

## Requirements

### Hardware Requirements
Minimum 8GB RAM
Minimum 2 cores CPU (4 cores recommended)
GPU (optional but recommended for faster processing)

### Software Requirements
Python 3.8 or above
Required Python Libraries:
Transformers (for BERT)
NLTK (for NLP tasks)
Scikit-learn (for cosine similarity and L2 normalization)
Streamlit (for the web interface)

### Operating System:
Windows, macOS, or Linux

## Installation
### Clone the repository:
git clone https://github.com/NIRMAL1508/Legal-Assistance-System-for-Criminal-Violence-Victims-using-Natural-Language-Processing.git

### Navigate to the project directory:
cd Legal-Assistance-System-for-Criminal-Violence-Victims-using-Natural-Language-Processing

### Install the required Python libraries:
pip install -r requirements.txt

### Run the Streamlit application:
streamlit run app.py

## Dataset
The system uses a custom dataset derived from legal documents and real-life incident reports. The dataset is tokenized and processed to create embeddings that are used for comparison with user queries.

## Model
The core of the system is a fine-tuned BERT model:

### Pre-training:
The model was pre-trained on large amounts of legal text to capture the nuances of legal language.
### Fine-tuning:
Fine-tuned using specific legal cases related to criminal violence to ensure relevance and accuracy.
### Embeddings: 
Word and sentence embeddings are generated to understand and match user inputs with relevant legal sections.

## Results
The model’s performance was evaluated using a top-10 accuracy method, where the model suggests the top 10 most relevant legal actions for a given user query:

### Top-5 Accuracy:
BERT with L2 Normalization: 32%
Tuned BERT with L2 Normalization: 56%
### Top-10 Accuracy:
BERT with L2 Normalization: 46%
Tuned BERT with L2 Normalization: 62%
These results demonstrate the system’s ability to provide accurate legal guidance to victims of criminal violence.

## Future Work
The system can be further improved by:

### Expanding the Dataset: 
Including more diverse legal cases to cover a broader range of criminal violence scenarios.
### Enhanced NLP Models: 
Experimenting with newer models and techniques for better performance.
### Multilingual Support: 
Extending the system to support multiple languages to cater to a broader audience.

## Contributors
Adarsh G

Elanthamil R

Jeevan Krishna K V

Nirmal M

Ajay Deepak P M

## License
This project is licensed under the MIT License - see the LICENSE file for details.
