# AssemblyAI Audio Analysis Toolkit

This project demonstrates how to build an audio analysis system powered by AssemblyAI and the Model Context Protocol (MCP).

We use the following tech stack:

- AssemblyAI for audio transcription and analysis (audio-RAG)
- Streamlit for the interactive web UI
- Cursor as the MCP host for programmatic access

## Setup and Installation

Ensure you have Python 3.12 or later installed on your system.

### Install dependencies

```bash
# Clone the repository and navigate to the project directory
# git clone <your-repo-url>
cd project-name

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# MacOS/Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configure environment variables
Copy `.env.example` to `.env` and configure the following environment variables:

```
ASSEMBLYAI_API_KEY=your_assemblyai_api_key
```

## Usage

### 1. Run as a Streamlit App (Interactive UI)

Launch the web app for interactive audio analysis:

```bash
streamlit run app.py
```

- **Upload Audio**: Drag and drop or browse for audio files (WAV, MP3, MP4, M4A, FLAC)
- **Processing**: The app automatically processes your audio with AssemblyAI
- **Analysis**: Navigate through different tabs to explore results:
  - View timestamped transcription
  - Read AI-generated summaries
  - Analyze speaker patterns
  - Explore sentiment analysis
  - Discover key topics
  - Chat with your audio content

## Author
### Gavara Lokesh
📧 lokeshgavara1@gmail.com
🔗 LinkedIn Profile

