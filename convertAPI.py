from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pyttsx3
import wave
import base64
import requests
import torch
import fitz
import logging as logger
from flask import Flask, request, jsonify
from googletrans import Translator
from google.cloud import speech
from google.api_core.client_options import ClientOptions
from transformers import pipeline
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from huggingface_hub import login
import pickle, cv2, time, threading
import numpy as np
import mediapipe as mp
import random
from dotenv import load_dotenv


load_dotenv()

HUGGINGFACE_API_KEY = os.getenv("HF_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
app = Flask(__name__)
CORS(app) 
# Example Unicode Braille patterns for some characters in different languages
# NOTE: This is simplified and partial. Add full mappings as needed.
login(token=HUGGINGFACE_API_KEY)  # Use your own HF token
recognition_thread_started = False
API_KEY = GOOGLE_API_KEY  # Your Google API key
PDF_PATH =  os.getenv("FILE_PATH")
braille_mappings = {
    "Tamil": {
        'அ': '⠁',
    'ஆ': '⠜',
    'இ': '⠊',
    'ஈ': '⠔',
    'உ': '⠥',
    'ஊ': '⠳',
    'எ': '⠢',
    'ஏ': '⠑',
    'ஐ': '⠌',
    'ஒ': '⠭',
    'ஓ': '⠕',
    'ஔ': '⠪',
    'க': '⠅',
    'ங': '⠬',
    'ச': '⠉',
    'ஞ': '⠒',
    'ட': '⠾',
    'ண': '⠼',
    'த': '⠞',
    'ந': '⠝',
    'ப': '⠏',
    'ம': '⠍',
    'ய': '⠽',
    'ர': '⠗',
    'ல': '⠇',
    'வ': '⠧',
    'ழ': '⠷',
    'ள': '⠸',
    'ற': '⠻',
    'ன': '⠰',
    'ஜ': '⠚',
    'ஷ': '⠯',
    'ஸ': '⠎',
    'ஹ': '⠓',
    '்': '⠈',
    'ஃ': '⠠',
    'ா': '⠜',
    'ி': '⠊',
    'ீ': '⠔',
    'ு': '⠥',
    'ூ': '⠳',
    'ெ': '⠢',
    'ே': '⠑',
    'ை': '⠌',
    'ொ': '⠭',
    'ோ': '⠪',
    'ௌ': '⠪',
    ' ': ' ',
    ',': '⠂',
    ';': '⠆',
    ':': '⠒',
    '!': '⠖',
    '?': '⠦',
    '.': '⠲'
        # add more characters...
    },
    "English": {
        'a': '⠁', 'b': '⠃', 'c': '⠉', 'd': '⠙', 'e': '⠑',
    'f': '⠋', 'g': '⠛', 'h': '⠓', 'i': '⠊', 'j': '⠚',
    'k': '⠅', 'l': '⠇', 'm': '⠍', 'n': '⠝', 'o': '⠕',
    'p': '⠏', 'q': '⠟', 'r': '⠗', 's': '⠎', 't': '⠞',
    'u': '⠥', 'v': '⠧', 'w': '⠺', 'x': '⠭', 'y': '⠽',
    'z': '⠵',
        # add more characters...
    },
    "Hindi": {
        'अ': '⠁',
    'आ': '⠅',
    'इ': '⠃',
    'ई': '⠊',
    'उ': '⠕',
    'ऊ': '⠎',
    'ऋ': '⠖',
    'ए': '⠋',
    'ऐ': '⠍',
    'ओ': '⠕',
    'औ': '⠕',
    'क': '⠉',
    'ख': '⠩',
    'ग': '⠛',
    'घ': '⠹',
    'ङ': '⠗',
    'च': '⠉',
    'छ': '⠘',
    'ज': '⠐',
    'झ': '⠖',
    'ञ': '⠝',
    'ट': '⠘',
    'ठ': '⠪',
    'ड': '⠐',
    'ढ': '⠘',
    'ण': '⠜',
    'त': '⠖',
    'थ': '⠝',
    'द': '⠘',
    'ध': '⠹',
    'न': '⠝',
    'प': '⠛',
    'फ': '⠋',
    'ब': '⠘',
    'भ': '⠩',
    'म': '⠝',
    'य': '⠽',
    'र': '⠉',
    'ल': '⠎',
    'व': '⠗',
    'श': '⠎',
    'ष': '⠖',
    'स': '⠘',
    'ह': '⠛',
    'क्ष': '⠛',
    'ज्ञ': '⠗'
        # add more characters...
    },
    "Telugu": {
       'అ': '⠁',
    'ఆ': '⠜',
    'ఇ': '⠃',
    'ఈ': '⠊',
    'ఉ': '⠕',
    'ఊ': '⠎',
    'ఎ': '⠋',
    'ఏ': '⠑',
    'ఐ': '⠡',
    'ఒ': '⠕',
    'ఓ': '⠕⠁',
    'ఔ': '⠥',
    'క': '⠅',
    'ఖ': '⠹',
    'గ': '⠛',
    'ఘ': '⠳',
    'ఙ': '⠬',
    'చ': '⠉',
    'ఛ': '⠡',
    'జ': '⠚',
    'ఝ': '⠚⠓',
    'ఞ': '⠝',
    'ట': '⠾',
    'ఠ': '⠾⠹',
    'డ': '⠙',
    'ఢ': '⠙⠓',
    'ణ': '⠻',
    'త': '⠞',
    'థ': '⠹',
    'ద': '⠙',
    'ధ': '⠙⠓',
    'న': '⠝',
    'ప': '⠏',
    'ఫ': '⠋',
    'బ': '⠃',
    'భ': '⠃⠓',
    'మ': '⠍',
    'య': '⠽',
    'ర': '⠗',
    'ల': '⠇',
    'వ': '⠧',
    'శ': '⠩',
    'ష': '⠩⠓',
    'స': '⠎',
    'హ': '⠓',
    'ళ': '⠭',
    'క్ష': '⠅⠩',
    'జ్ఞ': '⠚⠝',
    'ా': '⠜',
    'ి': '⠃',
    'ీ': '⠊',
    'ు': '⠕',
    'ూ': '⠎',
    'ె': '⠋',
    'ే': '⠑',
    'ై': '⠡',
    'ొ': '⠕',
    'ో': '⠕⠁',
    'ౌ': '⠥',
    '౦': '⠴',
    '౧': '⠂',
    '౨': '⠆',
    '౩': '⠒',
    '౪': '⠲',
    '౫': '⠢',
    '౬': '⠖',
    '౭': '⠶',
    '౮': '⠦',
    '౯': '⠔',
    '।': '⠲',
    '॥': '⠶',
    ',': '⠂',
    '.': '⠲',
    '?': '⠦',
    '!': '⠖'
        # add more characters...
    },
    "Marathi": {
        'अ': '⠁',
    'आ': '⠜',
    'इ': '⠊',
    'ई': '⠔',
    'उ': '⠕',
    'ऊ': '⠾',
    'ऋ': '⠗',
    'ए': '⠑',
    'ऐ': '⠜⠊',
    'ओ': '⠕',
    'औ': '⠕⠕',
    'क': '⠅',
    'ख': '⠩',
    'ग': '⠛',
    'घ': '⠹',
    'ङ': '⠝',
    'च': '⠉',
    'छ': '⠡',
    'ज': '⠚',
    'झ': '⠯',
    'ञ': '⠻',
    'ट': '⠾',
    'ठ': '⠞',
    'ड': '⠹',
    'ढ': '⠽',
    'ण': '⠻',
    'त': '⠹',
    'थ': '⠹⠹',
    'द': '⠙',
    'ध': '⠹⠙',
    'न': '⠝',
    'प': '⠏',
    'फ': '⠋',
    'ब': '⠃',
    'भ': '⠃⠃',
    'म': '⠍',
    'य': '⠽',
    'र': '⠗',
    'ल': '⠇',
    'ळ': '⠸',
    'व': '⠺',
    'श': '⠩',
    'ष': '⠯',
    'स': '⠎',
    'ह': '⠓',
    'ा': '⠜',
    'ि': '⠊',
    'ी': '⠔',
    'ु': '⠕',
    'ू': '⠾',
    'ृ': '⠗',
    'े': '⠑',
    'ै': '⠜⠊',
    'ो': '⠕',
    'ौ': '⠕⠕',
    'ं': '⠴',
    'ः': '⠦',
    '्': ''  
        # add more characters...
    },
    "Kannada": {
       'ಅ': '⠁',
    'ಆ': '⠜',
    'ಇ': '⠊',
    'ಈ': '⠔',
    'ಉ': '⠕',
    'ಊ': '⠾',
    'ಋ': '⠗',
    'ಎ': '⠑',
    'ಏ': '⠣',
    'ಐ': '⠜⠊',
    'ಒ': '⠕',
    'ಓ': '⠕⠜',
    'ಔ': '⠕⠕',
    'ಕ': '⠅',
    'ಖ': '⠩',
    'ಗ': '⠛',
    'ಘ': '⠹',
    'ಙ': '⠝',
    'ಚ': '⠉',
    'ಛ': '⠡',
    'ಜ': '⠚',
    'ಝ': '⠯',
    'ಞ': '⠻',
    'ಟ': '⠾',
    'ಠ': '⠞',
    'ಡ': '⠹',
    'ಢ': '⠽',
    'ಣ': '⠻',
    'ತ': '⠹',
    'ಥ': '⠹⠹',
    'ದ': '⠙',
    'ಧ': '⠹⠙',
    'ನ': '⠝',
    'ಪ': '⠏',
    'ಫ': '⠋',
    'ಬ': '⠃',
    'ಭ': '⠃⠃',
    'ಮ': '⠍',
    'ಯ': '⠽',
    'ರ': '⠗',
    'ಲ': '⠇',
    'ಳ': '⠭',
    'ವ': '⠺',
    'ಶ': '⠩',
    'ಷ': '⠯',
    'ಸ': '⠎',
    'ಹ': '⠓',
    'ಾ': '⠜',
    'ಿ': '⠊',
    'ೀ': '⠔',
    'ು': '⠕',
    'ೂ': '⠾',
    'ೃ': '⠗',
    'ೆ': '⠑',
    'ೇ': '⠣',
    'ೈ': '⠜⠊',
    'ೊ': '⠕',
    'ೋ': '⠕⠜',
    'ೌ': '⠕⠕',
    'ಂ': '⠴',
    'ಃ': '⠦',
    '್': ''  
        # add more characters...
    },
    "Malayalam": {
       'ക': '⠅',
    'ഖ': '⠩',
    'ഗ': '⠛',
    'ഘ': '⠹',
    'ങ': '⠬',
    'ച': '⠉',
    'ഛ': '⠡',
    'ജ': '⠚',
    'ഝ': '⠯',
    'ഞ': '⠻',
    'ട': '⠾',
    'ഠ': '⠹',
    'ഡ': '⠙',
    'ഢ': '⠭',
    'ണ': '⠻',
    'ത': '⠞',
    'ഥ': '⠹',
    'ദ': '⠙',
    'ധ': '⠧',
    'ന': '⠝',
    'ഩ': '⠝⠝',
    'പ': '⠏',
    'ഫ': '⠋',
    'ബ': '⠃',
    'ഭ': '⠞',
    'മ': '⠍',
    'യ': '⠽',
    'ര': '⠗',
    'റ': '⠻',
    'ല': '⠇',
    'ള': '⠳',
    'ഴ': '⠮',
    'വ': '⠺',
    'ശ': '⠩',
    'ഷ': '⠯',
    'സ': '⠎',
    'ഹ': '⠓',
    'ഺ': '⠞⠞⠞',
    '൦': '⠚',
    '൧': '⠁',
    '൨': '⠃',
    '൩': '⠉',
    '൪': '⠙',
    '൫': '⠑',
    '൬': '⠋',
    '൭': '⠛',
    '൮': '⠓',
    '൯': '⠊',
    '.': '⠲',
    ',': '⠂',
    '।': '⠦',
    '॥': '⠖',
    'ഽ': '⠄',
    '൹': '⠼'
        # add more characters...
    },
    "Bengali": {
       'অ': '⠁',
    'আ': '⠜',
    'ই': '⠊',
    'ঈ': '⠔',
    'উ': '⠕',
    'ঊ': '⠾',
    'ঋ': '⠗',
    'এ': '⠑',
    'ঐ': '⠜⠊',
    'ও': '⠕',
    'ঔ': '⠕⠕',
    'ক': '⠅',
    'খ': '⠩',
    'গ': '⠛',
    'ঘ': '⠹',
    'ঙ': '⠝',
    'চ': '⠉',
    'ছ': '⠡',
    'জ': '⠚',
    'ঝ': '⠯',
    'ঞ': '⠻',
    'ট': '⠾',
    'ঠ': '⠞',
    'ড': '⠹',
    'ঢ': '⠽',
    'ণ': '⠻',
    'ত': '⠹',
    'থ': '⠹⠹',
    'দ': '⠙',
    'ধ': '⠹⠙',
    'ন': '⠝',
    'প': '⠏',
    'ফ': '⠋',
    'ব': '⠃',
    'ভ': '⠃⠃',
    'ম': '⠍',
    'য': '⠽',
    'র': '⠗',
    'ল': '⠇',
    'শ': '⠩',
    'ষ': '⠯',
    'স': '⠎',
    'হ': '⠓',
    'া': '⠜',
    'ি': '⠊',
    'ী': '⠔',
    'ু': '⠕',
    'ূ': '⠾',
    'ৃ': '⠗',
    'ে': '⠑',
    'ৈ': '⠜⠊',
    'ো': '⠕',
    'ৌ': '⠕⠕',
    'ং': '⠴',
    'ঃ': '⠦',
    '্': ''  
    }
}

@app.route('/convert', methods=['POST'])
def convert_to_braille():
    data = request.json
    text = data.get('text', '')
    language = data.get('language', '')

    if not text or not language:
        return jsonify({'error': 'Missing text or language parameter'}), 400

    mapping = braille_mappings.get(language)
    if mapping is None:
        return jsonify({'error': f'Language "{language}" not supported'}), 400

    braille_result = ''
    for char in text:
        braille_char = mapping.get(char, char)  # fallback to same char if not mapped
        braille_result += braille_char

    return jsonify({'braille': braille_result})
def translate_to_english(text: str):
    return Translator().translate(text, dest="en").text

def translate_to_hindi(text: str):
    return Translator().translate(text, dest="hi").text

def translate_to_tamil(text: str):
    return Translator().translate(text, dest="ta").text

# ---------------- Speech to Text ----------------

def transcribe_audio(filename: str, lang_code: str = "en-IN"):
    client = speech.SpeechClient(client_options=ClientOptions(api_key=API_KEY))
    with open(filename, "rb") as audio_file:
        content = audio_file.read()
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code=lang_code,
    )
    response = client.recognize(config=config, audio=audio)
    return " ".join([result.alternatives[0].transcript for result in response.results])

# ---------------- PDF + QA ----------------

def extract_pdf_text(pdf_path: str):
    doc = fitz.open(pdf_path)
    return "".join([page.get_text("text") for page in doc])

def split_text(text: str, chunk_size=1000, chunk_overlap=100):
    return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap).split_text(text)

def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    documents = [Document(page_content=chunk) for chunk in chunks]
    return FAISS.from_documents(documents, embedding=embeddings)

def similarity_search_by_query(vector_store, query, k=5):
    return [result.page_content for result in vector_store.similarity_search(query, k=k)]

def run_qa(documents, query):
    qa_pipeline = pipeline("question-answering", model="deepset/roberta-large-squad2", device=0 if torch.cuda.is_available() else -1)
    answers = []
    for doc in documents:
        try:
            result = qa_pipeline({'context': doc, 'question': query})
            answers.append(result['answer'])
        except Exception as e:
            logger.error(f"QA Error: {e}")
    return " ".join(answers) if answers else "No answer found."

# Load PDF and prepare vector store once at startup
pdf_text = extract_pdf_text(PDF_PATH)
chunks = split_text(pdf_text)
vector_store = create_vector_store(chunks)

# ---------------- Endpoints ----------------

@app.route('/process-audio', methods=['POST'])
def process_audio():
    if 'file' not in request.files or 'lang' not in request.form:
        return jsonify({"error": "Missing audio file or language code"}), 400

    file = request.files['file']
    lang_code = request.form['lang']
    audio_path = "input.wav"
    file.save(audio_path)

    try:
        input_text = transcribe_audio(audio_path, lang_code)
        if lang_code != "en-IN":
            input_text = translate_to_english(input_text)

        results = similarity_search_by_query(vector_store, input_text, k=10)
        rag_output = run_qa(results, input_text)

        # Translate back to original language
        if lang_code == "hi-IN":
            response_text = translate_to_hindi(rag_output)
        elif lang_code == "ta-IN":
            response_text = translate_to_tamil(rag_output)
        else:
            response_text = rag_output

        return jsonify({
            "input_text": input_text,
            "response_text": response_text
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get("text")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        results = similarity_search_by_query(vector_store, question, k=10)
        answer = run_qa(results, question)
        print(f"Question: {question}, Answer: {answer}")
        return jsonify({"response": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, max_num_hands=1)
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M',
    13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
    25: 'Z', 26: '0', 27: '1', 28: '2', 29: '3', 30: '4', 31: '5', 32: '6', 33: '7', 34: '8', 35: '9',
    36: ' ',
    37: '.'
}
current_alphabet = "N/A"
current_word = ""
current_sentence = ""
expected_features = 42

cap = cv2.VideoCapture(0)
stabilization_buffer = []
last_registered_time = time.time()
registration_delay = 1.5

def recognize_sign():
    global current_alphabet, current_word, current_sentence
    global stabilization_buffer, last_registered_time

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                data_aux = []
                x_ = []
                y_ = []

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                if len(data_aux) < expected_features:
                    data_aux.extend([0] * (expected_features - len(data_aux)))
                elif len(data_aux) > expected_features:
                    data_aux = data_aux[:expected_features]

                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]

                stabilization_buffer.append(predicted_character)
                if len(stabilization_buffer) > 30:
                    stabilization_buffer.pop(0)

                if stabilization_buffer.count(predicted_character) > 25:
                    current_time = time.time()
                    if current_time - last_registered_time > registration_delay:
                        last_registered_time = current_time
                        current_alphabet = predicted_character

                        if predicted_character == ' ':
                            if current_word.strip():
                                current_sentence += current_word + " "
                            current_word = ""
                        elif predicted_character == '.':
                            if current_word.strip():
                                current_sentence += current_word + "."
                            current_word = ""
                        else:
                            current_word += predicted_character

@app.route("/start-sign", methods=["POST"])
def start_sign_recognition():
    global recognition_thread_started
    if not recognition_thread_started:
        recognition_thread_started = True
        threading.Thread(target=recognize_sign, daemon=True).start()
        return jsonify({"message": "Sign recognition started"}), 200
    else:
        return jsonify({"message": "Recognition already running"}), 200


@app.route('/get-sign-result', methods=['GET'])
def get_sign_result():
    return jsonify({
        "alphabet": current_alphabet,
        "word": current_word,
        "sentence": current_sentence
    })



LETTERS = [chr(i) for i in range(65, 91)]  # A-Z

@app.route('/get-question', methods=['GET'])
def get_question():
    global current_question, result
    current_question = random.choice(list(labels_dict.values()))
    result = None
    return jsonify({"question_letter": current_question})

@app.route('/check-result', methods=['GET'])
def check_result():
    return jsonify({
        "predicted": current_prediction,
        "result": result if result else "Waiting..."
    })
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
