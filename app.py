"""
Sandra - Voice AI Q&A Interview System
----------------------------------------
A Streamlit app that asks a fixed set of questions in the user's chosen
language (English or Hindi), reads them aloud, accepts typed or
live-spoken answers via a built-in mic, and stores the full session as a
local JSON file.

Voice details:
  * English questions are spoken with Groq's natural TTS (Orpheus).
  * Hindi questions use a free neural voice via edge-tts.
  * Live answer dictation uses the browser's Web Speech API in the
    selected language.
"""

import asyncio
import base64
import json
import os
import uuid
from datetime import datetime

import edge_tts
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from groq import Groq

# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------

load_dotenv()


def _load_api_key() -> str | None:
    """Read the Groq key from the environment (.env locally) or, when
    deployed on Streamlit Community Cloud, from st.secrets."""
    key = os.getenv("GROQ_API_KEY")
    if key:
        return key
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return None


GROQ_API_KEY = _load_api_key()

APP_NAME = "Sandra"
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Each supported language carries its own translated questions, the BCP-47
# code used for browser speech recognition + synthesis, and a few UI strings.
LANGUAGES = {
    "English": {
        "code": "en",
        "speech_lang": "en-US",
        "intro": "Hi, I'm Sandra. I'll ask you {n} short questions. "
                 "Answer by typing or by speaking with the mic.",
        "placeholder": "Type or click the mic to speak, then press Enter",
        "questions": [
            "What is your full name?",
            "What is your age?",
            "What is your current occupation?",
            "What city do you live in?",
            "What is your highest qualification?",
        ],
    },
    "हिन्दी (Hindi)": {
        "code": "hi",
        "speech_lang": "hi-IN",
        "edge_voice": "hi-IN-SwaraNeural",   # free neural Hindi voice (edge-tts)
        "intro": "नमस्ते, मैं सैंड्रा हूँ। मैं आपसे {n} छोटे सवाल पूछूँगी। "
                 "आप टाइप करके या माइक से बोलकर जवाब दे सकते हैं।",
        "placeholder": "टाइप करें या माइक पर क्लिक करके बोलें, फिर Enter दबाएँ",
        "questions": [
            "आपका पूरा नाम क्या है?",
            "आपकी उम्र क्या है?",
            "आपका वर्तमान व्यवसाय क्या है?",
            "आप किस शहर में रहते हैं?",
            "आपकी सर्वोच्च योग्यता क्या है?",
        ],
    },
}


def current_lang() -> dict:
    """Return the config dict for the language chosen for this session."""
    return LANGUAGES[st.session_state.lang]


def get_questions() -> list:
    """Return the question list in the session's chosen language."""
    return current_lang()["questions"]

st.set_page_config(page_title=APP_NAME, page_icon="🎤", layout="centered")

# Custom answer box: a text field with a built-in mic for live dictation.
# It returns {text, count, method} to Python only when the user presses Enter.
_mic_input = components.declare_component(
    "mic_input",
    path=os.path.join(PROJECT_DIR, "mic_input_component"),
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def generate_speech(text: str) -> bytes | None:
    """Generate natural-sounding speech audio for text using Groq TTS."""
    if not GROQ_API_KEY:
        st.warning("GROQ_API_KEY not found. Skipping voice playback.")
        return None

    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.audio.speech.create(
            model="canopylabs/orpheus-v1-english",
            voice="hannah",
            input=text,
            response_format="wav",
        )
        return response.read()
    except Exception as exc:
        # TTS is a nice-to-have; never block the flow if the call fails.
        st.warning(f"Could not generate voice audio: {exc}")
        return None


def edge_tts_speech(text: str, voice: str) -> bytes:
    """Generate neural speech audio using edge-tts (free, no API key).

    edge-tts streams MP3 chunks from Microsoft Edge's online voices.
    """
    async def _generate() -> bytes:
        communicate = edge_tts.Communicate(text, voice)
        buffer = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buffer += chunk["data"]
        return buffer

    return asyncio.run(_generate())


def play_audio_hidden(audio_bytes: bytes, mime: str = "audio/wav") -> None:
    """Autoplay audio in the browser with NO visible player.

    An <audio> element without the `controls` attribute renders nothing,
    so the voice is heard but the player UI stays invisible. The audio is
    embedded inline as base64 so no temp file/URL is needed.
    """
    b64 = base64.b64encode(audio_bytes).decode()
    html = f"""
        <audio autoplay style="display:none">
            <source src="data:{mime};base64,{b64}" type="{mime}">
        </audio>
    """
    st.markdown(html, unsafe_allow_html=True)


def speak_browser(text: str, speech_lang: str) -> None:
    """Speak text using the browser's built-in speech synthesis.

    Used for languages Groq's English-only TTS can't handle (Hindi, Punjabi).
    The browser picks a matching installed voice for `speech_lang`.
    """
    payload = json.dumps(text)  # safely escape for embedding in JS
    components.html(
        f"""
        <script>
          const text = {payload};
          const u = new SpeechSynthesisUtterance(text);
          u.lang = "{speech_lang}";
          u.rate = 0.95;
          function speak() {{
              const voices = window.speechSynthesis.getVoices();
              // Prefer a voice whose language matches the requested one.
              const match = voices.find(v => v.lang && v.lang.toLowerCase()
                                  .startsWith("{speech_lang}".slice(0, 2).toLowerCase()));
              if (match) u.voice = match;
              window.speechSynthesis.cancel();
              window.speechSynthesis.speak(u);
          }}
          if (window.speechSynthesis.getVoices().length) speak();
          else window.speechSynthesis.onvoiceschanged = speak;
        </script>
        """,
        height=0,
    )


def speak_question(text: str, lang: dict) -> None:
    """Read a question aloud in the session's language.

    Voice routing, best free option per language:
      * English -> Groq's natural TTS (Orpheus).
      * Hindi   -> edge-tts neural voice (free, no key).
    """
    if lang["code"] == "en":
        audio_bytes = generate_speech(text)
        if audio_bytes:
            play_audio_hidden(audio_bytes, "audio/wav")
    elif lang.get("edge_voice"):
        try:
            audio_bytes = edge_tts_speech(text, lang["edge_voice"])
            play_audio_hidden(audio_bytes, "audio/mpeg")
        except Exception as exc:
            # If edge-tts can't reach the service, fall back to browser voice.
            st.warning(f"Neural voice unavailable, using browser voice: {exc}")
            speak_browser(text, lang["speech_lang"])
    else:
        speak_browser(text, lang["speech_lang"])


def init_state() -> None:
    """Initialize all session_state keys used by the app."""
    defaults = {
        "started": False,
        "lang": "English",   # chosen interview language
        "current_index": 0,
        "responses": [],
        "session_id": str(uuid.uuid4()),
        "spoken_index": -1,   # tracks which question index has already been spoken aloud
        "completed": False,
        "saved_path": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_session() -> None:
    """Reset all state to start a brand new interview session."""
    st.session_state.started = False
    st.session_state.current_index = 0
    st.session_state.responses = []
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.spoken_index = -1
    st.session_state.completed = False
    st.session_state.saved_path = None
    # `lang` is intentionally left as-is so the picker keeps the last choice.


def save_responses() -> str:
    """Save the collected responses to a timestamped JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"responses_{timestamp}.json"
    filepath = os.path.join(PROJECT_DIR, filename)

    data = {
        "session_id": st.session_state.session_id,
        "timestamp": datetime.now().isoformat(),
        "language": st.session_state.lang,
        "responses": st.session_state.responses,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return filepath


def submit_answer(answer: str, input_method: str) -> None:
    """Record the answer for the current question and advance the flow."""
    if not answer:
        st.warning("Please provide an answer before submitting.")
        return

    questions = get_questions()
    question = questions[st.session_state.current_index]
    st.session_state.responses.append(
        {"question": question, "answer": answer, "input_method": input_method}
    )
    st.session_state.current_index += 1

    if st.session_state.current_index >= len(questions):
        st.session_state.completed = True
        st.session_state.saved_path = save_responses()

    st.rerun()


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------

init_state()

st.title(f"🎤 {APP_NAME}")
st.caption("Your Voice AI Q&A interviewer")

# ---- Landing screen --------------------------------------------------
if not st.session_state.started:
    # Language preference: questions are asked and answered in this language.
    st.session_state.lang = st.radio(
        "Choose your language / भाषा चुनें",
        options=list(LANGUAGES.keys()),
        index=list(LANGUAGES.keys()).index(st.session_state.lang),
        horizontal=True,
    )

    lang = current_lang()
    st.write(lang["intro"].format(n=len(lang["questions"])))

    if lang["code"] == "hi":
        st.caption("ℹ️ Hindi uses a free neural voice (edge-tts) — needs internet.")

    if st.button("Start Interview", type="primary"):
        st.session_state.started = True
        st.rerun()

# ---- Session complete screen -----------------------------------------
elif st.session_state.completed:
    st.success("✅ Session Complete! Thank you for your answers.")

    st.subheader("Chat History")
    for item in st.session_state.responses:
        with st.chat_message("assistant"):
            st.write(item["question"])
        with st.chat_message("user"):
            st.write(f"{item['answer']}  _(via {item['input_method']})_")

    st.subheader("Saved JSON")
    full_data = {
        "session_id": st.session_state.session_id,
        "timestamp": datetime.now().isoformat(),
        "language": st.session_state.lang,
        "responses": st.session_state.responses,
    }
    st.json(full_data)

    # On cloud hosts the local file is ephemeral, so offer a download too.
    st.download_button(
        "⬇️ Download responses (JSON)",
        data=json.dumps(full_data, indent=2, ensure_ascii=False),
        file_name=os.path.basename(st.session_state.saved_path or "responses.json"),
        mime="application/json",
    )

    if st.button("Start New Session", type="primary"):
        reset_session()
        st.rerun()

# ---- Active question flow ---------------------------------------------
else:
    lang = current_lang()
    questions = lang["questions"]
    idx = st.session_state.current_index
    total = len(questions)
    current_question = questions[idx]

    # Progress bar
    st.progress(idx / total, text=f"Question {idx + 1} of {total}")

    # Chat-style history of previously answered questions
    if st.session_state.responses:
        st.subheader("History")
        for item in st.session_state.responses:
            with st.chat_message("assistant"):
                st.write(item["question"])
            with st.chat_message("user"):
                st.write(f"{item['answer']}  _(via {item['input_method']})_")

    st.subheader(f"Question {idx + 1}")
    with st.chat_message("assistant"):
        st.write(current_question)

    # Speak the current question aloud the first time it's shown.
    if st.session_state.spoken_index != idx:
        speak_question(current_question, lang)
        st.session_state.spoken_index = idx

    # --- Your Answer: type OR speak live, press Enter to submit ---
    st.markdown("**Your Answer**")
    result = _mic_input(
        key=f"answer_{idx}",
        lang=lang["speech_lang"],
        placeholder=lang["placeholder"],
        default=None,
    )

    # The component only returns a value when Enter is pressed. We use the
    # incrementing `count` to make sure each submission fires exactly once.
    if isinstance(result, dict) and result.get("text"):
        seen_key = f"answer_count_{idx}"
        if result.get("count", 0) > st.session_state.get(seen_key, 0):
            st.session_state[seen_key] = result["count"]
            submit_answer(result["text"].strip(), result.get("method", "typed"))
