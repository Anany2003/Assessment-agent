import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import google.generativeai as genai
import PyPDF2 # For PDF parsing (optional, see notes)
from werkzeug.utils import secure_filename
import tempfile
import re
from flask_cors import CORS # Import Flask-CORS
import json # Import json for parsing question_types

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app) # Enable CORS for all origins on all routes

# Configure Google Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY not found in .env file.")
    exit(1)
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Gemini models
# Using gemini-1.5-flash for faster responses, can switch to gemini-1.5-pro for higher quality
model = genai.GenerativeModel('gemini-1.5-flash')

# Function to parse PDF (optional implementation)
def parse_pdf(file_path):
    """
    Parses a PDF file and extracts text content.
    Returns the extracted text.
    """
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text() or ""
            return text
    except Exception as e:
        print(f"Error parsing PDF: {e}")
        return None

@app.route('/')
def home():
    return "Assessment Agent Backend is running!"

@app.route('/generate_assessment', methods=['POST'])
def generate_assessment():
    """
    Generates assessment questions (MCQs and subjective) based on topic or PDF content.
    """
    topic = request.form.get('topic')
    total_questions = int(request.form.get('total_questions', 10)) # Default to 10 if not provided
    difficulty = request.form.get('difficulty', 'medium') # hard, medium, basic
    question_types_str = request.form.get('question_types', '["both"]') # Get question types as JSON string

    try:
        question_types = json.loads(question_types_str)
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid question_types parameter."}), 400

    syllabus_file = request.files.get('syllabus_pdf') # Files are accessed via request.files

    context_content = ""
    base_prompt_instruction = ""

    if syllabus_file:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            syllabus_file.save(tmp_file.name)
            temp_file_path = tmp_file.name
        
        context_content = parse_pdf(temp_file_path)
        os.unlink(temp_file_path) # Clean up the temporary file

        if not context_content:
            return jsonify({"error": "Failed to parse PDF syllabus."}), 500
        # CRITICAL CHANGE: Instruct the AI to derive questions FROM the syllabus content, not ABOUT it.
        base_prompt_instruction = "Derive questions from the content and topics described within the following syllabus. Do NOT ask questions ABOUT the syllabus document (e.g., about its structure, objectives count, or sections)."
    elif topic:
        base_prompt_instruction = f"Generate questions for the topic \"{topic}\"."
    else:
        return jsonify({"error": "Please provide a topic or upload a syllabus PDF."}), 400

    num_mcqs = 0
    num_subjective = 0
    mcqs_generated_text = ""
    subjective_generated_text = ""

    # Determine question counts based on selected question types
    if "mcq" in question_types and "subjective" in question_types:
        # Both selected: 2/3 MCQs, 1/3 subjective
        num_mcqs = round(total_questions * 2 / 3)
        num_subjective = total_questions - num_mcqs
        # Ensure at least 1 of each if total_questions > 0
        if total_questions > 0:
            if num_mcqs == 0: num_mcqs = 1
            if num_subjective == 0: num_subjective = 1
            if (num_mcqs + num_subjective) > total_questions: # Readjust if sums exceed due to rounding
                if num_mcqs > num_subjective:
                    num_mcqs = total_questions - num_subjective
                else:
                    num_subjective = total_questions - num_mcqs
    elif "mcq" in question_types:
        # Only MCQ selected
        num_mcqs = total_questions
        num_subjective = 0
    elif "subjective" in question_types:
        # Only Subjective selected
        num_subjective = total_questions
        num_mcqs = 0
    else:
        # Fallback, though frontend should prevent this
        return jsonify({"error": "No valid question type selected."}), 400


    try:
        if num_mcqs > 0:
            # --- MCQ Generation ---
            mcq_prompt = f"""
            {base_prompt_instruction}
            Generate exactly {num_mcqs} Multiple Choice Questions (MCQs) with "{difficulty}" difficulty.
            Each question MUST be clearly numbered (e.g., "Q1.", "Q2.").
            For each MCQ, provide EXACTLY 4 options (A, B, C, D) and clearly indicate the correct answer on a new line.
            Ensure questions are directly relevant to the subject matter of the topic/syllabus.

            Format each MCQ strictly as follows:
            Qn. Question text here.
            A. Option A text
            B. Option B text
            C. Option C text
            D. Option D text
            Correct Answer: [Letter, e.g., C]

            Example:
            Q1. What is the capital of France?
            A. Berlin
            B. Madrid
            C. Paris
            D. Rome
            Correct Answer: C

            Syllabus Content (if applicable, use this content to formulate questions related to its subject matter):
            {context_content if syllabus_file else "N/A"}
            """
            mcq_response = model.generate_content(mcq_prompt)
            mcqs_generated_text = mcq_response.text

        if num_subjective > 0:
            # --- Subjective Question Generation ---
            subjective_prompt = f"""
            {base_prompt_instruction}
            Generate exactly {num_subjective} subjective/descriptive questions with "{difficulty}" difficulty.
            Each question MUST be clearly numbered (e.g., "Q1.", "Q2.").
            For each subjective question, provide a concise rubric or key points for evaluation.
            Ensure questions are directly relevant to the subject matter of the topic/syllabus.

            Format each subjective question and its rubric strictly as follows:
            Qn. Question text here.
            Rubric: Key point 1, Key point 2, ...

            Example:
            Q1. Explain the concept of recursion in programming.
            Rubric: Base case, Recursive step, Stack usage, Examples

            Syllabus Content (if applicable, use this content to formulate questions related to its subject matter):
            {context_content if syllabus_file else "N/A"}
            """
            subjective_response = model.generate_content(subjective_prompt)
            subjective_generated_text = subjective_response.text

        return jsonify({
            "mcqs": mcqs_generated_text,
            "subjective_questions": subjective_generated_text
        })
    except Exception as e:
        print(f"Error generating assessment: {e}")
        return jsonify({"error": "Failed to generate assessment. Please try again."}), 500


@app.route('/evaluate_answer', methods=['POST'])
def evaluate_answer():
    """
    Evaluates a given answer (MCQ, subjective) and provides concise feedback.
    This endpoint is now designed to handle a single answer evaluation for flexibility,
    while the frontend will orchestrate batch evaluation.
    """
    data = request.json
    question_text = data.get('question_text')
    user_answer = data.get('user_answer')
    question_type = data.get('question_type') # 'mcq', 'subjective'
    correct_answer = data.get('correct_answer') # For MCQs, this would be 'A', 'B', etc.
    rubric = data.get('rubric') # For subjective questions

    # Check for missing question text or question type which are always essential
    if not all([question_text, question_type]):
        return jsonify({"error": "Missing essential question parameters for evaluation."}), 400

    evaluation_result = {"score": "N/A", "feedback": "No evaluation performed.", "skill_gap": "N/A", "solution_highlights": "N/A"}

    if question_type == 'mcq':
        # Generate solution highlights for incorrect/unanswered MCQs
        if not user_answer or user_answer.strip().upper() != correct_answer.strip().upper():
            explanation_prompt = f"""
            For the MCQ: "{question_text}"
            The correct answer is {correct_answer}.
            Provide a short (1-2 sentences), clear explanation for why {correct_answer} is the correct answer.
            """
            try:
                explanation_response = model.generate_content(explanation_prompt)
                evaluation_result["solution_highlights"] = explanation_response.text.strip()
            except Exception as e:
                print(f"Error generating MCQ explanation: {e}")
                evaluation_result["solution_highlights"] = "Could not generate solution explanation due to an AI error."

        if not user_answer:
            evaluation_result["score"] = "Incorrect"
            evaluation_result["feedback"] = f"No answer provided."
            evaluation_result["skill_gap"] = "Incomplete Answer"
        elif user_answer.strip().upper() == correct_answer.strip().upper():
            evaluation_result["score"] = "Correct"
            evaluation_result["feedback"] = "Your answer is correct!"
            evaluation_result["skill_gap"] = "N/A"
            evaluation_result["solution_highlights"] = "N/A" # No solution needed if correct
        else:
            evaluation_result["score"] = "Incorrect"
            evaluation_result["feedback"] = f"Your answer is incorrect."
            evaluation_result["skill_gap"] = "Conceptual Understanding"

    elif question_type == 'subjective':
        # Treat unanswered subjective question as 0/10
        if not user_answer or user_answer.strip() == "":
            evaluation_result["score"] = "0/10"
            evaluation_result["feedback"] = "No answer provided for this subjective question."
            evaluation_result["skill_gap"] = "No Answer Provided"
            # Generate a concise solution/key points when no answer is provided
            solution_prompt = f"""
            Provide a short (1-3 sentences) and clear summary of the key points for a correct answer to the following question, based on the rubric:
            Question: {question_text}
            Rubric: {rubric if rubric else "N/A"}
            """
            try:
                solution_response = model.generate_content(solution_prompt)
                evaluation_result["solution_highlights"] = solution_response.text.strip()
            except Exception as e:
                print(f"Error generating solution for unanswered subjective question: {e}")
                evaluation_result["solution_highlights"] = "Could not generate solution highlights due to an AI error."
        else:
            subjective_eval_prompt = f"""
            Question: {question_text}
            User's Answer: {user_answer}
            Rubric/Key Points for evaluation: {rubric if rubric else "N/A"}

            Evaluate the user's answer for the subjective question based on the question and the provided rubric/key points.
            Provide a concise score (e.g., 7/10).
            Then, provide brief, direct, constructive feedback (1-3 sentences) on key strengths, weaknesses, and what was missed.
            Clearly state ONE most significant skill gap identified.
            Finally, provide concise 'Correct Approach/Solution Highlights' (1-3 sentences) detailing what a good answer would include.

            Format your response strictly as follows:
            Score: [Score, e.g., 7/10]
            Feedback: [Brief feedback]
            Skill Gap: [Most significant skill gap]
            Correct Approach/Solution Highlights: [Concise key points for correct answer]
            """
            try:
                eval_response = model.generate_content(subjective_eval_prompt)
                raw_eval_text = eval_response.text
                
                score_match = re.search(r"Score:\s*(.*?)\n", raw_eval_text)
                if score_match:
                    evaluation_result["score"] = score_match.group(1).strip()
                else:
                    evaluation_result["score"] = "N/A"

                feedback_match = re.search(r"Feedback:\s*(.*?)(?=\nSkill Gap:|$)", raw_eval_text, re.DOTALL)
                if feedback_match:
                    evaluation_result["feedback"] = feedback_match.group(1).strip()
                else:
                    evaluation_result["feedback"] = "Could not parse detailed feedback from AI."

                skill_gap_match = re.search(r"Skill Gap:\s*(.*?)(?=\nCorrect Approach/Solution Highlights:|$)", raw_eval_text, re.DOTALL)
                if skill_gap_match:
                    evaluation_result["skill_gap"] = skill_gap_match.group(1).strip()
                else:
                    evaluation_result["skill_gap"] = "Could not parse skill gap analysis from AI."
                
                solution_highlights_match = re.search(r"Correct Approach/Solution Highlights:\s*(.*)", raw_eval_text, re.DOTALL)
                if solution_highlights_match:
                    evaluation_result["solution_highlights"] = solution_highlights_match.group(1).strip()
                else:
                    evaluation_result["solution_highlights"] = "Could not parse solution highlights from AI."

            except Exception as e:
                print(f"Error evaluating subjective answer: {e}")
                evaluation_result["feedback"] = "Failed to evaluate subjective answer due to an AI error."
                evaluation_result["raw_gemini_response"] = str(e)


    elif question_type == 'coding':
        # Placeholder for coding test evaluation (requires execution environment/test cases)
        evaluation_result["score"] = "N/A"
        evaluation_result["feedback"] = "Coding test evaluation is a complex feature that requires a secure execution environment and test cases. This is a placeholder."
        evaluation_result["skill_gap"] = "Not applicable for placeholder coding evaluation."
        evaluation_result["solution_highlights"] = "N/A"

    else:
        return jsonify({"error": "Invalid question type."}), 400

    return jsonify(evaluation_result)


@app.route('/check_plagiarism', methods=['POST'])
def check_plagiarism():
    """
    Placeholder for plagiarism checker.
    A real plagiarism checker would compare the user's answer against a vast database.
    For demonstration, it can compare against a set of known answers or use Gemini for similarity.
    """
    data = request.json
    user_answer = data.get('user_answer')
    known_corpus = data.get('known_corpus', []) # A list of other answers or texts to compare against

    if not user_answer:
        return jsonify({"error": "No answer provided for plagiarism check."}), 400

    # Basic plagiarism check using Gemini for similarity (conceptual)
    if known_corpus:
        comparison_text = "\n".join(known_corpus)
        plagiarism_prompt = f"""
        User's Answer: "{user_answer}"
        Other Texts for comparison: "{comparison_text}"

        Analyze the user's answer and the other texts.
        Determine if the user's answer shows significant similarity or direct copying from any of the other texts.
        Provide a plagiarism score (e.g., 0-100%, 0% being no plagiarism, 100% being direct copy) and a brief explanation.
        Be cautious and state if the similarity is coincidental or common phrasing.

        Format:
        Plagiarism Score: [Score]%
        Explanation: [Detailed explanation]
        """
        try:
            plagiarism_response = model.generate_content(plagiarism_prompt)
            return jsonify({"plagiarism_report": plagiarism_response.text})
        except Exception as e:
            print(f"Error checking plagiarism: {e}")
            return jsonify({"error": "Failed to check plagiarism due to AI error."}), 500
    else:
        return jsonify({"plagiarism_report": "No corpus provided for comparison. Cannot perform a detailed plagiarism check."})


@app.route('/recommend_tests', methods=['POST'])
def recommend_tests():
    """
    Placeholder for skill gap analyzer and test recommender.
    This would typically require a history of user performance and a curriculum structure.
    For now, it can recommend based on the identified skill gaps from the last evaluation.
    """
    data = request.json
    skill_gaps = data.get('skill_gaps', []) # From previous evaluations
    user_profile = data.get('user_profile', {}) # e.g., {'current_level': 'intermediate', 'learning_style': 'visual'}

    if not skill_gaps:
        return jsonify({"recommendations": "No specific skill gaps identified to recommend new tests."})

    recommendation_prompt = f"""
    Based on the following identified skill gaps: {', '.join(skill_gaps)}.
    And the user's profile: {user_profile}.

    Recommend specific topics, types of assessments (e.g., more practice MCQs, hands-on projects, detailed essays),
    or learning resources (e.g., "Review Chapter 3 on Algorithms", "Practice more data structure problems") to address these gaps.
    Provide actionable and relevant recommendations.

    Recommendations:
    - [Recommendation 1]
    - [Recommendation 2]
    ...
    """
    try:
        recommendation_response = model.generate_content(recommendation_prompt)
        return jsonify({"recommendations": recommendation_response.text})
    except Exception as e:
        print(f"Error recommending tests: {e}")
        return jsonify({"error": "Failed to generate recommendations due to AI error."}), 500


if __name__ == '__main__':
    app.run(debug=True) # Set debug=False in production
