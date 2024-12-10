import openai
import pandas as pd
import json
import pdfplumber
import os
from docx import Document
from dotenv import load_dotenv
import gradio as gr

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

def extract_text_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    text = df.applymap(str).apply(lambda row: " | ".join(row), axis=1).str.cat(sep="\n")
    return text

def extract_text_from_json(json_path):
    with open(json_path, "r") as file:
        data = json.load(file)
    return json.dumps(data, indent=2)

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = "".join(page.extract_text() for page in pdf.pages if page.extract_text())
    return text

def extract_text_from_excel(excel_path):
    excel_data = pd.ExcelFile(excel_path)
    text = ""
    for sheet_name in excel_data.sheet_names:
        df = excel_data.parse(sheet_name)
        sheet_text = df.applymap(str).apply(lambda row: " | ".join(row), axis=1).str.cat(sep="\n")
        text += f"Sheet: {sheet_name}\n{sheet_text}\n"
    return text

def extract_text_from_word(word_path):
    doc = Document(word_path)
    text = "\n".join(paragraph.text for paragraph in doc.paragraphs if paragraph.text)
    return text

def extract_text(file):
    file_extension = os.path.splitext(file.name)[-1].lower()

    if file_extension == ".csv":
        return extract_text_from_csv(file.name)
    elif file_extension == ".json":
        return extract_text_from_json(file.name)
    elif file_extension == ".pdf":
        return extract_text_from_pdf(file.name)
    elif file_extension in [".xlsx", ".xls"]:
        return extract_text_from_excel(file.name)
    elif file_extension == ".docx":
        return extract_text_from_word(file.name)
    else:
        raise ValueError("Unsupported file format. Please upload a CSV, JSON, PDF, Excel, or Word file.")

def split_text_into_chunks(text, chunk_size=1500):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def process_query(file, user_query, conversation_history):
    try:
        if file.name not in conversation_history:
            extracted_text = extract_text(file)
            context_chunks = split_text_into_chunks(extracted_text)
            conversation_history[file.name] = context_chunks
        else:
            context_chunks = conversation_history[file.name]

        messages = [
            {"role": "system", "content": "You are an assistant that answers questions based on the uploaded file content."}
        ]
        for chunk in context_chunks:
            messages.append({"role": "system", "content": chunk})
        
        messages.append({"role": "user", "content": user_query})

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=150,  
            temperature=0.7
        )

        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Error: {str(e)}"
  
conversation_history = {}

def chat_interface(file, user_message, chatbot_state):
    if chatbot_state is None:
        chatbot_state = []
    
    response = process_query(file, user_message, conversation_history)
    
    chatbot_state.append(("User: " + user_message, "Assistant: " + response))
    return chatbot_state, chatbot_state

interface = gr.Interface(
    fn=chat_interface,
    inputs=[
        gr.File(label="Upload File (CSV, JSON, PDF, Excel, Word)"),
        gr.Textbox(label="Your Message"),
        gr.State()
        
    ],
    outputs=[
        gr.Chatbot(label="File Assistant Chatbot"),
        gr.State()
    ],
    title="chatbot assistant ",
    description="Upload a file (CSV, JSON, PDF, Excel, Word) and ask questions about its content in a chatbot-like interface."
)

if __name__ == "__main__":
    interface.launch()
