# File Assistant

File Assistant is a data assistant designed to help you with questions about your file's content. Whether you need information from a PDF, DOC, DOCX, or TXT file, File Assistant is here to assist you.

## Features

- Upload files in .pdf, .doc, .docx, or .txt format.
- Ask questions about the content of your uploaded file.
- Receive answers and insights based on the content of your file.

## Usage
1. Install the dependencies by
```commandline
pip install -r requirements.txt
```
Make sure you have `pip` installed. If not, you can install it by following the instructions [here](https://pip.pypa.io/en/stable/installation/).

2. Run the server by
```commandline
python server.py
```
This command assumes that your main server file is named `server.py` and is located in the current directory. Adjust the command as needed if your file is located elsewhere or named differently

3. A web should be opened automatically. If not, open your browser and go to `http://localhost:8080/`.
4. Once you see the site with chatbot named File Assistant, you can upload file and ask questions.

## Supported Formats

File Assistant currently supports files in .pdf, .doc, .docx, or .txt format.
