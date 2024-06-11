import logging
import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import worker  # Import the worker module
# Some of the code are from Coursera Lab https://www.coursera.org/learn/building-gen-ai-powered-applications/
# Modified and tested by Wenting Zheng

# Initialize Flask app and CORS
app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
app.logger.setLevel(logging.ERROR)

# Define the route for the index page
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')  # Render the index.html template

# Define the route for processing messages
@app.route('/process-message', methods=['POST'])
def process_message_route():
    user_message = request.json['userMessage']  # Extract the user's message from the request
    print('user_message', user_message)

    bot_response = worker.process_prompt(user_message)  # Process the user's message using the worker module

    # Return the bot's response as JSON
    return jsonify({
        "botResponse": bot_response
    }), 200

# Define the route for processing documents
@app.route('/process-document', methods=['POST'])
def process_document_route():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({
            "botResponse": "It seems like the file was not uploaded correctly, can you try "
                           "again. If the problem persists, try using a different file"
        }), 400

    file = request.files['file']  # Extract the uploaded file from the request

    file_path = file.filename  # Define the path where the file will be saved
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension != '.pdf' and file_extension != '.doc' and file_extension != '.docx' and file_extension != '.txt':
        return jsonify({
            "botResponse": "We currently do not support files in this format. Please upload files in .pdf, .doc, .docx, .txt, or .csv format."
        }), 400

    file.save(file_path)  # Save the file
    worker.process_document(file_path, file_extension)  # Process the document using the worker module
    
    os.remove(file_path)
    
    # Return a success message as JSON
    return jsonify({
        "botResponse": "Thank you for providing your document. I have analyzed it, so now you can ask me any questions regarding it!"
    }), 200

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, port=8000, host='0.0.0.0')
