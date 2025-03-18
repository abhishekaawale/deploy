from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import mysql.connector as mycon
from datetime import timedelta
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import os

app = Flask(__name__)
app.secret_key = "your_secret_key"
app.permanent_session_lifetime = timedelta(days=7)

# Database Connection
mydb = mycon.connect(host="localhost", user="root", password="Admin@123", database="signup")
db_cur = mydb.cursor()

# Load PDFs from the directory
pdf_directory = "C:\\Users\\Abhishek Awale\\Desktop\\msbte navigator demo\\data demo"
loader = PyPDFDirectoryLoader(pdf_directory)
documents = loader.load()

# Split text into smaller chunks
text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=20)
texts = text_splitter.split_documents(documents)

# Summarization using Facebook's Bart-large CNN model
summarizer_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
summarizer_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
summarizer_pipe = pipeline("summarization", model=summarizer_model, tokenizer=summarizer_tokenizer)

summarized_texts = []
for text in texts:
    if len(text.page_content) > 100:
        summary = summarizer_pipe(
            text.page_content,
            max_length=150,
            min_length=30,
            do_sample=False
        )[0]['summary_text']
        summarized_texts.append(summary)
    else:
        summarized_texts.append(text.page_content)

# Create embeddings & knowledge base using FAISS
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
db = FAISS.from_texts(summarized_texts, embeddings)

# Load FLAN-T5 model for Q&A
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float32,
    device=0 if device == "cuda" else -1
)
llm = HuggingFacePipeline(pipeline=pipe)

# Set up Retrieval Q&A chain
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever())

@app.route('/')
def home():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    
    db_cur.execute("SELECT * FROM users WHERE username=%s AND password=%s", (username, password))
    result = db_cur.fetchone()
    
    if result:
        session['username'] = username
        flash("Login Successful!", "success")
        return redirect(url_for('index'))
    else:
        flash("Invalid Username or Password!", "danger")
        return redirect(url_for('home'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']  # Added email from form
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        if password != confirm_password:
            flash("Passwords do not match!", "danger")
            return redirect(url_for('signup'))
        
        try:
            # Ensure your 'users' table has a column for 'email'
            db_cur.execute(
                "INSERT INTO users (username, email, password) VALUES (%s, %s, %s)",
                (username, email, password)
            )
            mydb.commit()
            flash("Signup successful! You can now log in.", "success")
            return redirect(url_for('home'))
        except Exception as e:
            flash(f"Database Error: {e}", "danger")
    
    return render_template('signup.html')

@app.route('/index')
def index():
    if 'username' in session:
        return render_template('index.html', username=session['username'])
    return redirect(url_for('home'))

@app.route('/ask', methods=['POST'])
def ask():
    if 'username' not in session:
        return jsonify({"answer": "Please log in to use the chatbot."})
    
    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"answer": "Please enter a valid question."})
    
    answer = qa.run(question)
    return jsonify({"answer": answer if answer.strip() else "Not found in resources."})

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash("Logged out successfully!", "info")
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
