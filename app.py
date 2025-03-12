from flask import Flask, request, render_template
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    job_description = request.form['job_description']
    # Call DeepSeek-V3 to generate resume and cover letter
    resume, cover_letter = generate_resume_and_cover_letter(job_description)
    # Save files
    save_files(resume, cover_letter)
    return "Resume and cover letter generated successfully!"

def generate_resume_and_cover_letter(job_description):
    # Add your DeepSeek-V3 logic here
    return "Generated Resume", "Generated Cover Letter"

def save_files(resume, cover_letter):
    # Save files to a folder
    with open("output/resume.txt", "w") as f:
        f.write(resume)
    with open("output/cover_letter.txt", "w") as f:
        f.write(cover_letter)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)