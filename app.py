# ---------------------------
# 📌 Install Required Packages
# ---------------------------
#change the !pip to pip if executing on a terminal
!pip install -q sentence-transformers faiss-cpu transformers gtts deep-translator speechrecognition gradio

# ---------------------------
# 📌 Imports
# ---------------------------
import os
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from deep_translator import GoogleTranslator
from gtts import gTTS
import gradio as gr
import speech_recognition as sr
import tempfile

# ---------------------------
# 📌 Load Bhagavad Gita Dataset
# ---------------------------
df = pd.read_csv("/kaggle/input/bhagwad-gita-dataset/Bhagwad_Gita.csv")
df.columns = df.columns.str.lower()
df['full_text'] = (
    "Chapter " + df['chapter'].astype(str) +
    " Verse " + df['verse'].astype(str) + ": " +
    df['engmeaning']
)

# ---------------------------
# 📌 Generate Embeddings + FAISS
# ---------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df['full_text'].tolist(), show_progress_bar=True)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# ---------------------------
# 📌 Load Local LLM
# ---------------------------
llm_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
llm = AutoModelForCausalLM.from_pretrained(llm_model_name, device_map="auto")
chat_pipeline = pipeline("text-generation", model=llm, tokenizer=tokenizer)

# ---------------------------
# 📌 Main Function: Krishna GPT with TTS + Voice Input
# ---------------------------
def gita_gpt_llm(text_input, audio_input, language):
    query = text_input.strip() if text_input else ""

    # If no text but audio, convert audio to text
    if not query and audio_input is not None:
        recognizer = sr.Recognizer()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_input)
            tmp_file.flush()
            audio_file = sr.AudioFile(tmp_file.name)
            with audio_file as source:
                audio_data = recognizer.record(source)
                try:
                    query = recognizer.recognize_google(audio_data)
                except:
                    return "❌ Could not understand the audio. Please try again.", None

    if not query:
        return "⚠️ Please enter text or record your voice to ask Krishna.", None

    # Semantic Search
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k=3)
    top_idx = I[0][0]
    top_row = df.iloc[top_idx]
    top_shloka = f"## 📖 Chapter {top_row['chapter']}, Verse {top_row['verse']}\n\n{top_row['engmeaning']}"
    context = "\n".join([df.iloc[idx]['full_text'] for idx in I[0]])

    # LLM Prompt
    prompt = (
        f"You are Krishna answering a devotee's question based on the Bhagavad Gita.\n\n"
        f"Question: {query}\n\n"
        f"Relevant Verses:\n{context}\n\n"
        f"Answer:"
    )
    result = chat_pipeline(
        prompt,
        max_new_tokens=250,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    llm_answer = result[0]['generated_text'].split("Answer:")[-1].strip()
    if not llm_answer:
        llm_answer = "_No answer generated. Please try rephrasing your question._"

    # Translate if Hindi
    if language.lower() == "hindi":
        llm_answer = GoogleTranslator(source='auto', target='hi').translate(llm_answer)
        top_shloka = GoogleTranslator(source='auto', target='hi').translate(top_shloka)

    final_response = f"{top_shloka}\n\n---\n\n## ✨ Answer:\n{llm_answer}"

    # TTS
    tts_lang = "hi" if language.lower() == "hindi" else "en"
    tts = gTTS(text=llm_answer, lang=tts_lang)
    audio_path = "answer.mp3"
    tts.save(audio_path)

    return final_response, audio_path

# ---------------------------
# 📌 Gradio Interface
# ---------------------------
gr.Interface(
    fn=gita_gpt_llm,
    inputs=[
        gr.Textbox(label="📜 Type your question to Krishna"),
        gr.Audio(type="filepath", label="🎙️ Or upload your voice question (WAV/MP3)"),
        gr.Radio(["English", "Hindi"], value="English", label="Choose Answer Language")
    ],
    outputs=[
        gr.Markdown(label="Answer"),
        gr.Audio(label="🔊 Listen to Krishna's Answer")
    ],
    title="📜 Gita GPT - Text + Voice 🤖",
    description="Speak or type your question to Krishna. Get the most relevant shloka and a spoken answer in Hindi or English."
).launch(share=True)
