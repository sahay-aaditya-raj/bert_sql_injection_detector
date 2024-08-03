from flask import Flask, request, render_template_string, session
import tensorflow as tf
from transformers import BertTokenizer
import json
import os

# Initialize the Flask application
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Replace with a secure key in production

# Define the path to the saved model and history file
MODEL_PATH = 'model'
HISTORY_FILE = 'query_history.json'

# Load the pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load the saved TensorFlow model
saved_model = tf.saved_model.load(MODEL_PATH)

def load_history():
    """
    Load the query history from the file.
    """
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []

def save_history(history):
    """
    Save the query history to the file.
    """
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f)

def predict_query(query):
    """
    Predicts whether the given SQL query is malicious or non-malicious.
    
    Args:
        query (str): The SQL query to be evaluated.
    
    Returns:
        str: Prediction result as 'Malicious' or 'Non-malicious'.
    """
    try:
        # Tokenize and encode the input query
        encodings = tokenizer(query, padding=True, truncation=True, max_length=128, return_tensors='tf')
        
        # Prepare inputs for the model
        inputs = {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'token_type_ids': tf.zeros_like(encodings['input_ids'])  # BERT model requires token_type_ids
        }
        
        # Perform the prediction
        infer = saved_model.signatures["serving_default"]
        outputs = infer(**inputs)
        
        # Extract the logits and determine the predicted label
        logits = outputs['logits']
        predicted_labels = tf.argmax(logits, axis=1).numpy()
        label = predicted_labels[0]
        
        # Return the result
        return 'Malicious' if label == 1 else 'Non-malicious'
    
    except Exception as e:
        return f"An error occurred: {e}"

@app.route('/', methods=['GET', 'POST'])
def index():
    # Load the history from the file on every request
    history = load_history()
    
    if request.method == 'POST':
        user_query = request.form.get('query')
        if user_query:
            result = predict_query(user_query)
            # Add the new query and result to the history
            history.append({'query': user_query, 'result': result})
            # Keep only the last 5 queries
            if len(history) > 5:
                history.pop(0)
            # Save the updated history to the file
            save_history(history)
        else:
            result = "Please enter an SQL query."
        return render_template_string(TEMPLATE, result=result, query=user_query, history=history)
    
    return render_template_string(TEMPLATE, result=None, query=None, history=history)

# HTML template for the web page
TEMPLATE = '''
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <title>SQL Query Classifier</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">
    <style>
        body { 
            background-color: #f4f7f6; 
            font-family: 'Roboto', sans-serif; 
        }
        .container { 
            margin-top: 50px; 
            max-width: 800px; 
        }
        .header { 
            text-align: center; 
            margin-bottom: 30px; 
        }
        .header h1 { 
            font-size: 2.5rem; 
            color: #333; 
        }
        .form-group textarea { 
            border-radius: 10px; 
            box-shadow: 0 2px 4px rgba(0,0,0,.1); 
            font-size: 1rem; 
            height: 150px; 
        }
        .btn-custom { 
            background-color: #28a745; 
            color: white; 
            border-radius: 50px; 
            border: none; 
            padding: 12px 30px; 
            font-size: 1.1rem; 
            transition: background-color 0.3s, transform 0.3s; 
        }
        .btn-custom:hover { 
            background-color: #218838; 
            transform: scale(1.05); 
        }
        .result { 
            margin-top: 30px; 
            padding: 20px; 
            border-radius: 10px; 
            box-shadow: 0 4px 8px rgba(0,0,0,.2); 
            background-color: #fff; 
        }
        .result h2 { 
            margin-top: 0; 
            font-size: 1.5rem; 
            color: #333; 
        }
        .history { 
            display: flex;
            flex-direction: column-reverse 
        }
        .history h2 { 
            font-size: 1.5rem; 
            color: #333; 
        }
        .history-item { 
            background-color: #fff; 
            padding: 10px; 
            border-radius: 5px; 
            box-shadow: 0 1px 3px rgba(0,0,0,.1); 
            margin-bottom: 10px; 
        }
        .footer { 
            margin-top: 40px; 
            text-align: center; 
            font-size: 0.9rem; 
            color: #666; 
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1><i class="fas fa-database"></i> SQL Query Classifier</h1>
            <p class="lead">Enter an SQL query to classify it as 'Malicious' or 'Non-malicious'.</p>
        </div>
        <form method="post">
            <div class="form-group">
                <label for="query">SQL Query:</label>
                <textarea id="query" name="query" class="form-control">{{ query }}</textarea>
            </div>
            <button type="submit" class="btn btn-custom btn-lg btn-block">Predict</button>
        </form>
        {% if result is not none %}
            <div class="result">
                <h2><i class="fas fa-check-circle"></i> Prediction Result</h2>
                <p><strong>Query:</strong> {{ query }}</p>
                <p><strong>Prediction:</strong> {{ result }}</p>
            </div>
        {% endif %}
        <h2 style="margin-top:30px"><i class="fas fa-history"></i> Recent Queries</h2>
        <div class="history">
            
            {% if history %}
                {% for entry in history %}
                    <div class="history-item">
                        <p><strong>Query:</strong> {{ entry.query }}</p>
                        <p><strong>Result:</strong> {{ entry.result }}</p>
                    </div>
                {% endfor %}
            {% else %}
                <p>No recent queries.</p>
            {% endif %}
        </div>
        <div class="footer">
            <p>&copy; 2024 SQL Query Classifier. All rights reserved.</p>
        </div>
    </div>
</body>
</html>
'''

if __name__ == "__main__":
    app.run(debug=True)
