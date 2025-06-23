import os
import re
import json
from flask import Flask, render_template, request, redirect, url_for

# For PDF and DOCX extraction
import PyPDF2
from docx import Document

# For summarization using Transformers (Bart)
from transformers import BartTokenizer, BartForConditionalGeneration
import torch

# For question generation using OpenAI
import openai

# Set your OpenAI API key (or better, use an environment variable)
#gpt key
# ----- Utility Functions -----

def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def extract_text_from_docx(file_path):
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

def extract_text_from_file(file_path, file_type="auto"):
    ext = os.path.splitext(file_path)[1].lower() if file_type == "auto" else file_type.lower()
    if ext == ".pdf":
        text = extract_text_from_pdf(file_path)
    elif ext == ".docx":
        text = extract_text_from_docx(file_path)
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        raise ValueError("Unsupported file format")
    return clean_text(text)

def clean_text(text):
    unwanted = ["INSTITUTE", "UNIVERSITY", "COLLEGE", "CHAPTER", "SECTION"]
    return "\n".join(
        line for line in text.split("\n")
        if not any(word in line.upper() for word in unwanted)
    )

# ----- Summarization Functions -----

# Load the model and tokenizer globally (this may take a little time on first run)
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

def summarize(text, max_length=300):
    inputs = tokenizer(text, max_length=1024, return_tensors="pt", truncation=True)
    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=4,
        max_length=max_length,
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def chunk_text(text, max_tokens=1024):
    words = text.split()
    chunks = []
    chunk = []
    for word in words:
        chunk.append(word)
        tokenized_len = len(tokenizer(" ".join(chunk), truncation=False)["input_ids"])
        if tokenized_len >= max_tokens:
            chunk.pop()  # Remove the word that caused overflow
            chunks.append(" ".join(chunk))
            chunk = [word]
    if chunk:
        chunks.append(" ".join(chunk))
    return chunks

def summarize_large_text(text):
    text_chunks = chunk_text(text, max_tokens=1024)
    # Limit to 5 chunks to prevent long processing times
    if len(text_chunks) > 5:
        text_chunks = text_chunks[:5]
    summaries = [summarize(chunk) for chunk in text_chunks]
    return " ".join(summaries)

# ----- Dummy Subject Identifier -----
def identify_subject(cleaned_text):
    # In a real implementation, use NLP techniques to infer the subject.
    return "General Knowledge"

# ----- Question Generation Function -----
def generate_questions(input_type, text_input, subject, difficulty, num_mcq, num_tf):
    prompt = f"""Generate {num_mcq} multiple choice questions (MCQs) and {num_tf} True/False questions about:
Subject: {subject}
Difficulty: {difficulty}
Text: {text_input[:3000]}

Format exactly like:

MCQs:
1. Question?
a) Option1
b) Option2
c) Option3
d) Option4
Answer: a

True/False:
1. Statement
Answer: True
"""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a quiz generator."},
            {"role": "user", "content": prompt}
        ]
    )
    
    content = response.choices[0].message.content

    # --- Parsing Logic ---
    mcq_pattern = r'\d+\.\s(.*?)\n[a-d]\)\s(.*?)\n[a-d]\)\s(.*?)\n[a-d]\)\s(.*?)\n[a-d]\)\s(.*?)\nAnswer:\s([a-dA-D])'
    tf_pattern = r'\d+\.\s(.*?)\nAnswer:\s(True|False)'

    mcqs = []
    for match in re.finditer(mcq_pattern, content, re.DOTALL):
        question, opt1, opt2, opt3, opt4, answer = match.groups()
        mcqs.append({
            "question": question.strip(),
            "options": [opt1.strip(), opt2.strip(), opt3.strip(), opt4.strip()],
            "answer": answer.strip().lower()
        })

    tfs = []
    for match in re.finditer(tf_pattern, content):
        statement, answer = match.groups()
        tfs.append({
            "question": statement.strip(),
            "answer": answer.strip().lower()
        })

    questions = {
        "subject": subject,
        "difficulty": difficulty,
        "mcq": mcqs,
        "tf": tfs
    }

    # Save questions to a JSON file in the static folder.
    with open("static/questions.json", "w") as f:
        json.dump(questions, f, indent=2)

    return questions

# ----- Flask App Configuration -----
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ----- Routes -----

@app.route('/')
def index():
    return render_template('input.html',
                           selected_input_type='pdf',
                           num_mcq=5,
                           num_tf=5,
                           difficulty='medium')

@app.route('/summarize', methods=['POST'])
def summarize_route():
    input_type = request.form.get('inputType', 'pdf')
    num_mcq = int(request.form.get('num_mcq', 5))
    num_tf = int(request.form.get('num_tf', 5))
    difficulty = request.form.get('difficulty', 'medium')
    text_content = request.form.get('text_content', '')
    file = request.files.get('file')

    if input_type == 'text':
        if not text_content.strip():
            return render_template('input.html',
                                   error="Please enter text content",
                                   selected_input_type=input_type,
                                   text_content=text_content,
                                   num_mcq=num_mcq,
                                   num_tf=num_tf,
                                   difficulty=difficulty)
        try:
            summary = summarize_large_text(text_content)
            return render_template('summary.html',
                                   summary=summary,
                                   cleaned_text=text_content,
                                   num_mcq=num_mcq,
                                   num_tf=num_tf,
                                   difficulty=difficulty,
                                   input_type=input_type)
        except Exception as e:
            return render_template('input.html',
                                   error=f"Error processing text: {str(e)}",
                                   selected_input_type=input_type,
                                   text_content=text_content,
                                   num_mcq=num_mcq,
                                   num_tf=num_tf,
                                   difficulty=difficulty)
    else:
        if not file or file.filename == '':
            return render_template('input.html',
                                   error="Please upload a file",
                                   selected_input_type=input_type,
                                   num_mcq=num_mcq,
                                   num_tf=num_tf,
                                   difficulty=difficulty)
        try:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext not in [".pdf", ".docx", ".txt"]:
                return render_template('input.html',
                                       error="Unsupported file type",
                                       selected_input_type=input_type,
                                       num_mcq=num_mcq,
                                       num_tf=num_tf,
                                       difficulty=difficulty)
            text = extract_text_from_file(filepath)
            summary = summarize_large_text(text)
            return render_template('summary.html',
                                   summary=summary,
                                   cleaned_text=text,
                                   num_mcq=num_mcq,
                                   num_tf=num_tf,
                                   difficulty=difficulty,
                                   input_type=input_type)
        except Exception as e:
            return render_template('input.html',
                                   error=f"File processing error: {str(e)}",
                                   selected_input_type=input_type,
                                   num_mcq=num_mcq,
                                   num_tf=num_tf,
                                   difficulty=difficulty)

@app.route('/generate', methods=['POST'])
def generate_quiz():
    try:
        cleaned_text = request.form.get('cleaned_text', '')
        num_mcq = int(request.form.get('num_mcq', 5))
        num_tf = int(request.form.get('num_tf', 5))
        difficulty = request.form.get('difficulty', 'medium')
        input_type = request.form.get('input_type', 'text')

        subject = identify_subject(cleaned_text)
        generate_questions(
            input_type=input_type,
            text_input=cleaned_text,
            subject=subject,
            difficulty=difficulty,
            num_mcq=num_mcq,
            num_tf=num_tf
        )
        return redirect(url_for('quiz'))
    except Exception as e:
        return f"Error generating questions: {str(e)}"

@app.route('/quiz')
def quiz():
    try:
        with open("static/questions.json", "r") as f:
            data = json.load(f)
        mcq_questions = data.get("mcq", [])
        tf_questions = data.get("tf", [])
        return render_template('quiz.html',
                               mcq_questions=mcq_questions,
                               tf_questions=tf_questions)
    except Exception as e:
        return f"Error loading quiz: {str(e)}"

@app.route('/result', methods=['POST'])
def result():
    try:
        with open("static/questions.json", "r") as f:
            data = json.load(f)
        # Combine MCQ and True/False questions
        questions = data.get("mcq", []) + data.get("tf", [])
        
        results = []
        score = 0
        user_answers = request.form.to_dict()
        for idx, question in enumerate(questions):
            user_answer = str(user_answers.get(f"q{idx}", "")).strip().lower()
            correct_answer = str(question["answer"]).strip().lower()
            is_correct = (user_answer == correct_answer)
            if is_correct:
                score += 1
            results.append({
                "question": question["question"],
                "options": question.get("options", []),
                "correct_answer": correct_answer,
                "user_answer": user_answer,
                "is_correct": is_correct,
                "type": "mcq" if "options" in question and question["options"] else "tf"
            })
        
        total = len(questions)
        percentage = (score / total) * 100 if total > 0 else 0

        if percentage >= 90:
            performance = "Outstanding! 🎉"
        elif percentage >= 75:
            performance = "Excellent! 😊"
        elif percentage >= 60:
            performance = "Good Job! 🙂"
        else:
            performance = "Keep Practicing! 💪"
        
        # Get time elapsed from hidden input
        time_taken = request.form.get('timeElapsed', '0')
        
        return render_template('result.html',
                               score=score,
                               total=total,
                               performance=performance,
                               results=results,
                               time_taken=time_taken)
    except Exception as e:
        return f"Error calculating results: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
