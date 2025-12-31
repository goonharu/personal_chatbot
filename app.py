from flask import Flask, jsonify, render_template, request
import sqlite3
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv
import markdown2 
import edge_tts
from io import BytesIO
from pydub import AudioSegment
from pydub.effects import normalize, compress_dynamic_range
import torch
from TTS.api import TTS
import os
import time
import whisper
from werkzeug.utils import secure_filename
import re
import random

# Load environment variables from .env file
load_dotenv()

# vars
type_tts = "edge"  # Using Edge TTS for speed (30x faster!)

if type_tts == "coqui":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

print("Loading whisper model...")
# Model options (accuracy vs speed):
# tiny    - Fastest, least accurate (~1GB RAM)
# base    - Fast, okay accuracy (~1GB RAM) [OLD]
# small   - Balanced, good accuracy (~2GB RAM) [UPGRADED]
# medium  - Slower, very accurate (~5GB RAM)
# large   - Slowest, best accuracy (~10GB RAM)
whisper_model = whisper.load_model("small")  # Upgraded from "base"
print("Whisper model loaded (small - improved accuracy).")

def preprocess_audio_for_stt(input_path, output_path):
    """
    Preprocess audio to improve STT accuracy
    - Normalize volume
    - Reduce noise
    - Optimize sample rate
    - Compress dynamic range
    """
    try:
        # Load audio
        audio = AudioSegment.from_file(input_path)
        
        # Convert to mono (stereo can cause issues)
        if audio.channels > 1:
            audio = audio.set_channels(1)
        
        # Set optimal sample rate for Whisper (16kHz)
        audio = audio.set_frame_rate(16000)
        
        # Normalize volume (make quiet audio louder)
        audio = normalize(audio)
        
        # Compress dynamic range (reduce difference between loud and quiet)
        audio = compress_dynamic_range(audio, threshold=-20.0, ratio=4.0)
        
        # Remove silence from start and end
        audio = audio.strip_silence(silence_thresh=-40, padding=100)
        
        # Export preprocessed audio
        audio.export(output_path, format="wav")
        print(f"✅ Audio preprocessed: {len(audio)}ms")
        return output_path
        
    except Exception as e:
        print(f"⚠️ Audio preprocessing failed: {e}, using original")
        return input_path


conn = sqlite3.connect('chat_messages.db', check_same_thread=False)
c = conn.cursor()

c.execute(''' CREATE TABLE IF NOT EXISTS messages
          (role TEXT, content TEXT)''')

conn.commit()

# Action to Emotion/Gesture mapping
def extract_actions(text):
    """
    Extracts action markers from text and maps them to emotions/gestures
    Returns: {
        'cleaned_text': str (text without action markers),
        'actions': list of {'emotion': str, 'gesture': str, 'position': int}
    }
    """
    import re
    
    # Action patterns and their mappings - UPGRADED based on real conversations
    action_map = {
        # === SIGHS - Multiple contexts ===
        r'\*?sighs?( contentedly| heavily)?\*?': {'emotion': 'sad', 'gesture': 'lookdown', 'intensity': 0.7},
        r'sighs? contentedly': {'emotion': 'happy', 'gesture': None, 'intensity': 0.6},  # Happy sigh
        
        # === SMILES - Various types ===
        r'\*?smiles?( warmly| gently| softly)?\*?': {'emotion': 'happy', 'gesture': None, 'intensity': 0.8},
        r'\*?smirks?\*?': {'emotion': 'happy', 'gesture': None, 'intensity': 0.7},
        
        # === LAUGHS - Different intensities ===
        r'\*?laughs?( nervously| softly)?\*?': {'emotion': 'happy', 'gesture': None, 'intensity': 0.9},
        r'laughs? nervously': {'emotion': 'worried', 'gesture': None, 'intensity': 0.6},  # Nervous laugh
        r'\*?giggles?\*?': {'emotion': 'happy', 'gesture': None, 'intensity': 0.85},
        r'\*?chuckles?\*?': {'emotion': 'happy', 'gesture': None, 'intensity': 0.7},
        
        # === CRYING & SADNESS ===
        r'\*?crying\*?': {'emotion': 'crying', 'gesture': 'lookdown', 'intensity': 0.9},
        r'\*?tears up\*?': {'emotion': 'crying', 'gesture': None, 'intensity': 0.8},
        r'gets? teary-eyed': {'emotion': 'crying', 'gesture': None, 'intensity': 0.85},
        r'\*?shudders?\*?': {'emotion': 'worried', 'gesture': None, 'intensity': 0.7},
        
        # === BLUSHING & EMBARRASSMENT ===
        r'\*?blushes?\*?': {'emotion': 'embarrassed', 'gesture': None, 'intensity': 0.75},
        
        # === FACIAL EXPRESSIONS ===
        r'\*?frowns?\*?': {'emotion': 'sad', 'gesture': None, 'intensity': 0.6},
        r'\*?grins?\*?': {'emotion': 'happy', 'gesture': None, 'intensity': 0.8},
        r'\*?winks?\*?': {'emotion': 'happy', 'gesture': None, 'intensity': 0.7},
        
        # === ADVERBS (how she speaks) ===
        r'\*?excitedly\*?': {'emotion': 'excited', 'gesture': 'greet', 'intensity': 0.9},
        r'\*?sadly\*?': {'emotion': 'sad', 'gesture': None, 'intensity': 0.7},
        r'\*?nervously\*?': {'emotion': 'worried', 'gesture': None, 'intensity': 0.6},
        r'\*?happily\*?': {'emotion': 'happy', 'gesture': None, 'intensity': 0.8},
        r'\*?angrily\*?': {'emotion': 'angry', 'gesture': None, 'intensity': 0.8},
        r'\*?warmly\*?': {'emotion': 'happy', 'gesture': None, 'intensity': 0.8},
        r'\*?gently\*?': {'emotion': 'happy', 'gesture': None, 'intensity': 0.7},
        r'\*?softly\*?': {'emotion': 'happy', 'gesture': None, 'intensity': 0.6},
        r'\*?empathetically\*?': {'emotion': 'serious', 'gesture': None, 'intensity': 0.7},
        
        # === PHYSICAL GESTURES ===
        r'\*?nods?\*?': {'emotion': None, 'gesture': 'nod', 'intensity': 0.5},
        r'\*?shakes? head\*?': {'emotion': None, 'gesture': 'shake', 'intensity': 0.5},
        r'\*?tilts? head\*?': {'emotion': None, 'gesture': 'think', 'intensity': 0.5},
        r'\*?looks? down\*?': {'emotion': 'sad', 'gesture': 'lookdown', 'intensity': 0.6},
        r'\*?looks? away\*?': {'emotion': 'embarrassed', 'gesture': 'lookdown', 'intensity': 0.5},
        r'\*?leans? (forward|in)\*?': {'emotion': None, 'gesture': 'greet', 'intensity': 0.5},
        r'\*?waves?\*?': {'emotion': 'happy', 'gesture': 'greet', 'intensity': 0.7},
        r'\*?claps?\*?': {'emotion': 'excited', 'gesture': None, 'intensity': 0.85},
        r'\*?shrugs?\*?': {'emotion': 'neutral', 'gesture': None, 'intensity': 0.4},
        
        # === BREATHING & PAUSES ===
        r'\*?takes? a? deep breaths?\*?': {'emotion': 'serious', 'gesture': None, 'intensity': 0.7},
        r'\*?pauses?\*?': {'emotion': None, 'gesture': 'think', 'intensity': 0.5},
        r'\*?gulps?\*?': {'emotion': 'worried', 'gesture': None, 'intensity': 0.7},
        
        # === THINKING ACTIONS ===
        r'\*?thinks?\*?': {'emotion': 'serious', 'gesture': 'think', 'intensity': 0.6},
        r'\*?ponders?\*?': {'emotion': 'serious', 'gesture': 'think', 'intensity': 0.6},
        r'\*?hesitates?\*?': {'emotion': 'worried', 'gesture': 'think', 'intensity': 0.6},
    }
    
    actions = []
    cleaned_text = text
    
    # Find all actions in order of appearance
    for pattern, mapping in action_map.items():
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            position = match.start()
            actions.append({
                'text': match.group(),
                'position': position,
                'emotion': mapping['emotion'],
                'gesture': mapping['gesture'],
                'intensity': mapping['intensity']
            })
    
    # Sort by position (order they appear in text)
    actions.sort(key=lambda x: x['position'])
    
    # Remove action markers from text
    for pattern in action_map.keys():
        cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE)
    
    # Clean up extra spaces and punctuation
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Multiple spaces to single
    cleaned_text = re.sub(r'\s+([.,!?])', r'\1', cleaned_text)  # Space before punctuation
    cleaned_text = re.sub(r'\.\.\.+', '...', cleaned_text)  # Multiple dots to ellipsis
    cleaned_text = cleaned_text.strip()
    
    return {
        'cleaned_text': cleaned_text,
        'actions': actions
    }

def combine_emotion_data(text_analysis, action_data):
    """
    Combines emotion from text analysis with actions extracted from markers
    Priority: Actions > Text keywords
    """
    # Start with base analysis from text
    result = {
        'emotion': text_analysis['emotion'],
        'gesture': text_analysis['gesture'],
        'intensity': text_analysis['intensity'],
        'actions': action_data['actions']  # All actions for sequential animation
    }
    
    # If we have actions, use the first/strongest one as primary
    if action_data['actions']:
        primary_action = action_data['actions'][0]  # Use first action
        
        # Override with action's emotion if present
        if primary_action['emotion']:
            result['emotion'] = primary_action['emotion']
            result['intensity'] = max(result['intensity'], primary_action['intensity'])
        
        # Override with action's gesture if present
        if primary_action['gesture']:
            result['gesture'] = primary_action['gesture']
    
    return result

# Emotion and Gesture Detection System
def analyze_emotion(text):
    """
    Analyzes text for emotion and gestures
    Returns: {
        "emotion": str,
        "gesture": str or None,
        "intensity": float (0-1),
        "actions": list of actions for sequential animation
    }
    """
    # First, extract action markers
    action_data = extract_actions(text)
    cleaned_text = action_data['cleaned_text']
    
    text_lower = cleaned_text.lower()
    
    # Emotion keywords with intensity weights - UPGRADED from conversation analysis
    emotion_patterns = {
        'happy': {
            'keywords': ['happy', 'glad', 'joy', 'wonderful', 'great', 'awesome', 'amazing', 
                        'excellent', 'fantastic', 'love', 'lovely', 'excited', 'yay', 'haha', 
                        'lol', 'ahahaha', 'sweetheart', 'darling', 'grateful', 'smile', 'brighten',
                        'pleasant', 'delightful', 'fun', 'enjoyed'],
            'intensity': 0.8
        },
        'excited': {
            'keywords': ['wow', 'omg', 'incredible', 'unbelievable', '!!!!', 'woah', 'yaaas',
                        'chills', 'goosebumps', 'masterful', 'hooked', 'wild ride'],
            'intensity': 0.9
        },
        'sad': {
            'keywords': ['sad', 'sorry', 'unfortunately', 'disappointed', 'upset', 
                        'depressed', 'unhappy', 'terrible', 'awful', 'bad', 'somber',
                        'tough', 'heavy', 'heartbroken', 'heartbreaking', 'distressing',
                        'poor things', 'struggling', 'pain'],
            'intensity': 0.7
        },
        'angry': {
            'keywords': ['angry', 'mad', 'furious', 'annoyed', 'frustrated', 'irritated',
                        'bothers me'],
            'intensity': 0.8
        },
        'surprised': {
            'keywords': ['surprise', 'shocked', 'unexpected', 'really?', 'no way', 'seriously?',
                        'surreal', 'never expected'],
            'intensity': 0.7
        },
        'worried': {
            'keywords': ['worried', 'concerned', 'anxious', 'nervous', 'afraid', 'scared', 
                        'uncertain', 'unsure', 'careful', 'heartbreaking', 'heebie-jeebies',
                        'unsettling', 'distressing', 'helplessness', 'trapped', 'skeptical',
                        'uncomfortable', 'disturbing'],
            'intensity': 0.6
        },
        'serious': {
            'keywords': ['important', 'serious', 'critical', 'must', 'need to', 'have to', 
                        'resonates', 'thought-provoking', 'philosophical', 'complex',
                        'raw', 'unapologetic', 'real', 'accurately', 'reflection',
                        'wisdom', 'experience'],
            'intensity': 0.7
        },
        'embarrassed': {
            'keywords': ['embarrassed', 'shy', 'awkward', 'blush', 'oops', 'surreal',
                        'never expected to be'],
            'intensity': 0.6
        },
        'crying': {
            'keywords': ['cry', 'crying', 'tears', 'sob', 'heartbroken', 'teary-eyed'],
            'intensity': 0.9
        }
    }
    
    # Gesture keywords
    gesture_patterns = {
        'nod': ['yes', 'yeah', 'right', 'correct', 'exactly', 'agree', 'absolutely', 
                'definitely', 'sure', 'of course'],
        'shake': ['no', 'nope', 'wrong', 'incorrect', 'disagree', "don't"],
        'think': ['hmm', 'well', 'let me think', 'wonder', 'perhaps', 'maybe', 'possibly'],
        'greet': ['hello', 'hi ', 'hey ', 'greetings', 'good morning', 'good afternoon'],
        'lookdown': ['sorry', 'apologize', 'my apologies', 'my bad', 'forgive me']
    }
    
    # Detect emotion from cleaned text
    detected_emotion = 'neutral'
    max_intensity = 0.5
    
    for emotion, data in emotion_patterns.items():
        for keyword in data['keywords']:
            if keyword in text_lower:
                if data['intensity'] > max_intensity:
                    detected_emotion = emotion
                    max_intensity = data['intensity']
                break
    
    # Detect gesture from cleaned text
    detected_gesture = None
    for gesture, keywords in gesture_patterns.items():
        for keyword in keywords:
            if keyword in text_lower:
                detected_gesture = gesture
                break
        if detected_gesture:
            break
    
    # Check for questions (tilt head)
    if '?' in cleaned_text and not detected_gesture:
        detected_gesture = 'think'
    
    # Check for exclamations (more intensity)
    exclamation_count = cleaned_text.count('!')
    if exclamation_count >= 2:
        max_intensity = min(1.0, max_intensity + 0.2)
    
    # Create base analysis
    base_analysis = {
        'emotion': detected_emotion,
        'gesture': detected_gesture,
        'intensity': max_intensity
    }
    
    # Combine with action data (actions take priority)
    final_result = combine_emotion_data(base_analysis, action_data)
    
    # Return cleaned text for TTS
    final_result['cleaned_text'] = cleaned_text
    
    return final_result

# Voice assistant

# Edge TTS with Japanese accent
async def synth_speech_edge(TEXT, temp_file):
    # Try standard voice first to test
    VOICE = "en-US-AriaNeural"  # Standard US English (reliable)
    # VOICE = "en-US-JennyMultilingualNeural"  # Switch back if AriaNeural works
    # VOICE = "ja-JP-NanamiNeural"  # Native Japanese option
    
    # Debug: Check what text we're trying to synthesize
    print(f"📝 Edge TTS Input: '{TEXT[:100]}...' ({len(TEXT)} chars)")
    print(f"🎤 Using voice: {VOICE}")
    
    if not TEXT or len(TEXT.strip()) == 0:
        print("❌ Error: Empty text provided to TTS")
        return None
    
    communicate = edge_tts.Communicate(TEXT, VOICE, rate="+10%")
    
    try:
        # Save directly to file (Edge TTS handles format internally)
        await communicate.save(temp_file)
        
        # Verify file was created and has content
        if os.path.exists(temp_file):
            file_size = os.path.getsize(temp_file)
            print(f"✅ Audio saved to: {temp_file} ({file_size} bytes)")
            if file_size > 0:
                return temp_file
            else:
                print(f"❌ Audio file is empty!")
                return None
        else:
            print(f"❌ Audio file was not created!")
            return None
            
    except Exception as e:
        print(f"❌ Edge TTS Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def clean_text_for_tts(text):
    """
    Clean text by removing markdown and special characters that might cause TTS issues
    """
    # Remove markdown bold/italic
    text = re.sub(r'\*\*?(.*?)\*\*?', r'\1', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text, max_length=500):
    """
    Split text into chunks that are safe for TTS processing
    Tries to split at sentence boundaries
    
    Args:
        text: Text to split
        max_length: Maximum characters per chunk (default 500, can be adjusted)
    """
    # Clean the text first
    text = clean_text_for_tts(text)
    
    # If text is short enough, return as-is
    if len(text) <= max_length:
        return [text]
    
    # Split by sentences
    sentences = re.split(r'([.!?]+\s+)', text)
    
    chunks = []
    current_chunk = ""
    
    for i in range(0, len(sentences), 2):
        sentence = sentences[i]
        punctuation = sentences[i + 1] if i + 1 < len(sentences) else ""
        full_sentence = sentence + punctuation
        
        # If adding this sentence would exceed max_length, start new chunk
        if len(current_chunk) + len(full_sentence) > max_length and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = full_sentence
        else:
            current_chunk += full_sentence
    
    # Add remaining text
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

async def call_generate(text, temp_file, tts=None):
    if type_tts == "edge":
        await synth_speech_edge(text, temp_file)
    else:
        # Clean and limit text length for Coqui TTS
        cleaned_text = clean_text_for_tts(text)
        
        # Coqui TTS has a limit of around 500 characters for stable generation
        # You can adjust this value: 250 (very safe), 500 (balanced), 800 (risky)
        MAX_LENGTH = 500
        
        if len(cleaned_text) > MAX_LENGTH:
            cleaned_text = cleaned_text[:MAX_LENGTH] + "..."
            print(f"Warning: Text truncated to {MAX_LENGTH} chars for TTS")
        
        try:
            if tts is None:
                raise ValueError("TTS model not initialized for Coqui mode")
            tts.tts_to_file(text=cleaned_text, speaker_wav="voice/voice.wav", language="en", file_path=temp_file)
        except Exception as e:
            print(f"TTS Error: {e}")
            # Fallback: try with even shorter text (half the max length)
            fallback_text = cleaned_text[:MAX_LENGTH//2] + "..."
            print(f"Retrying with shorter text ({MAX_LENGTH//2} chars)")
            tts.tts_to_file(text=fallback_text, speaker_wav="voice/voice.wav", language="en", file_path=temp_file)
    
    return temp_file

async def synthesize(text, filename):
    """
    Synthesize speech using configured TTS engine
    - Edge TTS: Fast, no chunking needed
    - Coqui TTS: Parallel chunking for long text
    """
    # clear static/audio/ folder
    for file in os.listdir("./static/audio"):
        os.remove("./static/audio/" + file)
    
    # Clean text
    cleaned_text = clean_text_for_tts(text)
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    temp_file = f"./static/audio/{filename}-{timestamp}.wav"
    
    # Edge TTS: No chunking needed, handles any length
    if type_tts == "edge":
        print(f"🎤 Using Edge TTS for {len(cleaned_text)} chars...")
        try:
            await call_generate(cleaned_text, temp_file)
            print(f"✨ Edge TTS completed!")
            return temp_file
        except Exception as e:
            print(f"❌ Error with Edge TTS: {e}")
            fallback_text = "I apologize, but there was an error generating speech."
            await call_generate(fallback_text, temp_file)
            return temp_file
    
    # Coqui TTS: Use chunking for long text
    # If short enough, use simple approach
    if len(cleaned_text) <= 500:
        try:
            return await call_generate(cleaned_text, temp_file)
        except Exception as e:
            print(f"Error in simple synthesize: {e}")
            fallback_text = "I apologize, but there was an error generating speech."
            return await call_generate(fallback_text, temp_file)
    
    # Long text: chunk it and generate in PARALLEL (Coqui only)
    print(f"📝 Text is {len(cleaned_text)} chars, using PARALLEL chunking...")
    chunks = chunk_text(cleaned_text, max_length=500)
    print(f"✂️ Split into {len(chunks)} chunks - generating in parallel...")
    
    chunk_files = []
    
    # Generate all chunks in parallel using asyncio
    async def generate_chunk(i, chunk):
        chunk_file = f"./static/audio/{filename}-chunk{i}-{timestamp}.wav"
        try:
            start_time = time.time()
            await call_generate(chunk, chunk_file)
            elapsed = time.time() - start_time
            print(f"✅ Chunk {i+1}/{len(chunks)} done in {elapsed:.1f}s ({len(chunk)} chars)")
            return (i, chunk_file)
        except Exception as e:
            print(f"❌ Error generating chunk {i}: {e}")
            return (i, None)
    
    # Generate all chunks simultaneously
    import asyncio
    tasks = [generate_chunk(i, chunk) for i, chunk in enumerate(chunks)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Sort by index and collect successful chunks
    successful_chunks = []
    for result in results:
        if isinstance(result, tuple) and result[1] is not None:
            successful_chunks.append(result)
    
    successful_chunks.sort(key=lambda x: x[0])  # Sort by index
    chunk_files = [chunk[1] for chunk in successful_chunks]
    
    # Check if we got any successful chunks
    if not chunk_files:
        print("⚠️ All chunks failed, using fallback")
        fallback_text = "I apologize, but my response was too long to speak properly."
        final_file = f"./static/audio/{filename}-{timestamp}.wav"
        return await call_generate(fallback_text, final_file)
    
    # Load and combine all audio segments
    print(f"🔗 Combining {len(chunk_files)} audio segments...")
    audio_segments = []
    for chunk_file in chunk_files:
        try:
            audio_segments.append(AudioSegment.from_wav(chunk_file))
        except Exception as e:
            print(f"Warning: Could not load {chunk_file}: {e}")
    
    combined = AudioSegment.empty()
    pause = AudioSegment.silent(duration=400)  # 400ms pause between chunks
    
    for i, segment in enumerate(audio_segments):
        combined += segment
        if i < len(audio_segments) - 1:
            combined += pause
    
    # Export combined audio
    final_file = f"./static/audio/{filename}-combined-{timestamp}.wav"
    combined.export(final_file, format="wav")
    print(f"🎵 Combined audio: {len(combined)}ms duration")
    
    # Clean up individual chunk files
    for chunk_file in chunk_files:
        try:
            if os.path.exists(chunk_file):
                os.remove(chunk_file)
        except:
            pass
    
    print(f"✨ Successfully generated full audio from {len(chunks)} chunks!")
    return final_file

system_prompt = """You are named Nari, a reliable, caring, friendly senior female friend who always look after me. 

Keep your responses concise and natural (2-4 sentences typically). If you need to explain something longer, break it into digestible parts. Speak naturally as if in a conversation, not like writing an essay."""

def getAnswer(role, text):
    # Insert the message into the database
    c.execute("INSERT INTO messages VALUES (?, ?)", (role, text))
    conn.commit()

    # Get the last 5 messages
    c.execute("SELECT * FROM messages order by rowid DESC LIMIT 5")

    previous_messages = [{"role": row[0], "content": row[1]} for row in c.fetchall()]

    # REVERSE
    previous_messages = list(reversed(previous_messages))

    # Separate system message from conversation (Claude API format)
    system_message = system_prompt
    conversation_messages = []
    
    for msg in previous_messages:
        if msg['role'] != 'system':
            conversation_messages.append(msg)
    
    # === CLAUDE 3.5 HAIKU - CHEAPEST & FASTEST ===
    # Get API key from .env file
    CLAUDE_API_KEY = os.getenv('ANTHROPIC_API_KEY')
    
    if not CLAUDE_API_KEY or CLAUDE_API_KEY == 'your-api-key-here':
        raise ValueError("Please set ANTHROPIC_API_KEY in your .env file")
    
    client = Anthropic(api_key=CLAUDE_API_KEY)
    
    response = client.messages.create(
        model="claude-3-5-haiku-20241022",  # CHEAPEST: $0.80/$4 per million tokens
        max_tokens=1024,
        temperature=0.7,
        system=system_message,
        messages=conversation_messages
    )
    
    # Extract response from Claude API format
    bot_reponse = response.content[0].text.strip()

    c.execute("INSERT INTO messages VALUES (?, ?)", ("assistant", bot_reponse))

    conn.commit()
    return bot_reponse

app = Flask(__name__)

UPLOAD_FOLDER = './temp_audio'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/echo', methods=['POST'])
def echo():
    data = request.json
    print(data)
    message = data.get('message')
    return jsonify({'FROM': 'Echobot', 'MESSAGE': message})

@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']

        if audio_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"audio_{timestamp}.wav")
        audio_file.save(temp_path)

        print(f"🎤 Processing audio file...")
        
        # Preprocess audio for better accuracy
        processed_path = os.path.join(app.config['UPLOAD_FOLDER'], f"processed_{timestamp}.wav")
        final_path = preprocess_audio_for_stt(temp_path, processed_path)

        print(f"📝 Transcribing...")

        # Transcribe with improved accuracy settings
        result = whisper_model.transcribe(
            final_path,  # Use preprocessed audio
            language='en',
            fp16=False,  # Use FP32 for better accuracy (slower but more precise)
            task='transcribe',  # Explicitly set task
            temperature=0.0,  # Lower temperature = more conservative/accurate
            best_of=5,  # Try 5 different approaches, pick best
            beam_size=5,  # Use beam search for better results
            patience=1.0,  # Higher patience for beam search
            condition_on_previous_text=True,  # Use context from previous text
            initial_prompt="This is a casual conversation about anime, manga, and daily life."  # Context hint
        )
        transcription = result['text'].strip()
        
        print(f"✅ Transcription: {transcription}")

        # Cleanup temp files
        os.remove(temp_path)
        if os.path.exists(processed_path):
            os.remove(processed_path)

        return jsonify({'transcription': transcription})
    except Exception as e:
        print(f"Error during transcription: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/chat', methods=['POST'])
async def chat():
    data = request.json
    message = getAnswer("user", data['message'])

    # Analyze emotion and gestures (includes action extraction)
    analysis = analyze_emotion(message)
    
    print(f"🎭 Emotion Analysis: {analysis['emotion']} | Gesture: {analysis['gesture']} | Intensity: {analysis['intensity']}")
    if analysis.get('actions'):
        print(f"🎬 Detected {len(analysis['actions'])} actions: {[a['text'] for a in analysis['actions']]}")

    # Use cleaned text (without action markers) for TTS
    text_for_tts = analysis.get('cleaned_text', message)
    name_wav = await synthesize(text_for_tts, "out")
    
    return jsonify({
        'FROM': 'Nari', 
        'MESSAGE': markdown2.markdown(message),  # Original text with actions for display
        'WAV': name_wav,
        'EMOTION': analysis['emotion'],
        'GESTURE': analysis['gesture'],
        'INTENSITY': analysis['intensity'],
        'ACTIONS': analysis.get('actions', [])  # All actions for sequential animation
    })

# history
@app.route('/history', methods=['GET'])
def history():
    c.execute("SELECT * FROM messages order by rowid")
    previous_messages = [{"role": row[0], "content": markdown2.markdown(row[1])} for row in c.fetchall()]
    return jsonify(previous_messages)

if __name__ == '__main__':
    app.run(debug=True)