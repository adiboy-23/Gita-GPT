# 📜 Gita GPT - AI-Powered Bhagavad Gita Assistant

An intelligent conversational AI application that provides answers to questions based on the Bhagavad Gita. This application combines semantic search, local language models, and text-to-speech capabilities to create an interactive experience with Krishna's wisdom.

- <img width="1334" alt="Screenshot 2025-06-21 at 09 54 36" src="https://github.com/user-attachments/assets/6baead3e-8862-486d-8194-c2dd6926328e" />

- <img width="1333" alt="Screenshot 2025-06-20 at 19 51 28" src="https://github.com/user-attachments/assets/841e3f8d-54bb-4418-9e36-4c3b9c14380a" />


## ✨ Features

- **🤖 AI-Powered Responses**: Uses TinyLlama-1.1B-Chat model for generating contextual answers
- **🔍 Semantic Search**: Finds the most relevant verses from the Bhagavad Gita using FAISS and sentence transformers
- **🎙️ Voice Input**: Speak your questions using speech recognition
- **🔊 Text-to-Speech**: Listen to Krishna's answers in audio format
- **🌐 Multi-language Support**: Get answers in English or Hindi
- **📱 Web Interface**: Beautiful Gradio-based web interface for easy interaction
- **📖 Verse Context**: Displays relevant chapter and verse information with each answer

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Sufficient RAM (recommended: 8GB+ for model loading)
- Internet connection for initial model downloads

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd Gita-GPT
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Bhagavad Gita dataset**
   
   The application expects a CSV file named `Bhagwad_Gita.csv` in the `/kaggle/input/bhagwad-gita-dataset/` directory. You'll need to:
   - Download the dataset from Kaggle or prepare your own CSV file
   - Update the file path in `app.py` line 25 to point to your dataset location
   - Ensure the CSV has columns: `chapter`, `verse`, `engmeaning`

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the web interface**
   
   Open your browser and navigate to the URL displayed in the terminal (usually `http://127.0.0.1:7860`)

## 📋 Requirements

The application requires the following Python packages:

- `sentence-transformers` - For generating text embeddings
- `faiss-cpu` - For efficient similarity search
- `transformers` - For loading and using the TinyLlama model
- `gtts` - For text-to-speech functionality
- `deep-translator` - For language translation
- `SpeechRecognition` - For voice input processing
- `gradio` - For the web interface
- `pandas` - For data manipulation
- `numpy` - For numerical operations
- `torch` - For PyTorch-based models

## 🎯 Usage

### Text Input
1. Type your question in the text box
2. Select your preferred language (English or Hindi)
3. Click submit to get Krishna's answer

### Voice Input
1. Click the audio upload button
2. Record or upload your voice question (WAV/MP3 format)
3. Select your preferred language
4. Submit to receive both text and audio responses

### Example Questions
- "What does Krishna say about karma?"
- "How should one deal with difficult situations?"
- "What is the path to self-realization?"
- "How to control the mind?"

## 🏗️ Architecture

The application follows this workflow:

1. **Data Loading**: Loads Bhagavad Gita verses from CSV
2. **Embedding Generation**: Creates vector embeddings for all verses using SentenceTransformer
3. **Index Creation**: Builds a FAISS index for fast similarity search
4. **Query Processing**: 
   - Converts voice input to text (if applicable)
   - Generates embedding for the user query
   - Finds most similar verses using semantic search
5. **Answer Generation**: Uses TinyLlama model to generate contextual answers
6. **Translation**: Translates to Hindi if requested
7. **TTS**: Converts answer to speech
8. **Response**: Returns formatted text and audio

## 🔧 Configuration

### Model Settings
- **Embedding Model**: `all-MiniLM-L6-v2` (fast and efficient)
- **LLM Model**: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (lightweight local model)
- **Search Results**: Top 3 most relevant verses
- **Max Tokens**: 250 for answer generation
- **Temperature**: 0.7 for balanced creativity

### File Structure
```
Gita-GPT/
├── app.py              # Main application file
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── Bhagwad_Gita.csv   # Dataset file 
```

## 🐛 Troubleshooting

### Common Issues

1. **Model Download Issues**
   - Ensure stable internet connection
   - Check available disk space (models are several GB)
   - Try running with `--trust-remote-code` flag if needed

2. **Memory Issues**
   - Close other applications to free up RAM
   - Consider using a smaller model if available
   - Use CPU-only mode if GPU memory is insufficient

3. **Audio Issues**
   - Ensure microphone permissions are granted
   - Check audio file format (WAV/MP3 supported)
   - Verify system audio drivers

4. **Dataset Issues**
   - Ensure CSV file path is correct
   - Verify CSV has required columns: `chapter`, `verse`, `engmeaning`
   - Check file encoding (UTF-8 recommended)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Bhagavad Gita dataset contributors
- Hugging Face for the TinyLlama model
- Sentence Transformers library
- Gradio for the web interface
- All open-source contributors whose work made this possible

## 📞 Support

If you encounter any issues or have questions, please:
1. Check the troubleshooting section above
2. Search existing issues in the repository
3. Create a new issue with detailed information about your problem

---

**May Krishna's wisdom guide you on your spiritual journey! 🙏** 
