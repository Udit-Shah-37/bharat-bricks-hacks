"""Streamlit entrypoint: chat-first multilingual legal assistant."""

from __future__ import annotations

from dotenv import load_dotenv
import logging
import os
import sys
from pathlib import Path

load_dotenv()
# Repo root on Databricks Repos / local clone
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd
import streamlit as st

from nyaya_dhwani.triage_engine import TRIAGE_SYSTEM_PROMPT
from nyaya_dhwani.triage_service import TriageService
from nyaya_dhwani.query_logger import log_query
from nyaya_dhwani.sarvam_client import (
    is_configured as sarvam_configured,
    speech_to_text_file,
    strip_markdown_for_tts,
    text_to_speech_wav_bytes,
    transcript_from_stt_response,
    translate_text,
)

logger = logging.getLogger(__name__)

TOPIC_SEEDS: dict[str, str] = {
    "Domestic violence": "My husband beats me regularly and threatens to throw me out, what can I do?",
    "Defective product": "I bought a defective phone online and the seller is not responding to my complaint",
    "RTI request": "I filed an RTI application to get my birth certificate but the officer denied it without reason",
    "Illegal eviction": "My landlord is trying to evict me illegally without any notice period",
    "Wrongful termination": "I was wrongfully terminated from my job without being given any notice or severance pay",
    "FIR / Police": "Someone stole my motorcycle from outside my house, what should I do to file a complaint?",
    "Property dispute": "My neighbour has built a wall encroaching on my property, what legal action can I take?",
    "Consumer fraud": "An online seller charged me money but never delivered the product and is not refunding",
}

SARVAM_LANGUAGES: list[tuple[str, str]] = [
    ("en", "English"),
    ("hi", "Hindi · हिन्दी"),
    ("bn", "Bengali"),
    ("te", "Telugu"),
    ("mr", "Marathi"),
    ("ta", "Tamil"),
    ("gu", "Gujarati"),
    ("kn", "Kannada"),
    ("ml", "Malayalam"),
    ("pa", "Punjabi"),
    ("or", "Odia"),
    ("ur", "Urdu"),
    ("as", "Assamese"),
]

# UI ISO-ish code -> BCP-47 for Mayura / STT hints (best-effort for ur/as)
UI_TO_BCP47: dict[str, str] = {
    "en": "en-IN",
    "hi": "hi-IN",
    "bn": "bn-IN",
    "te": "te-IN",
    "mr": "mr-IN",
    "ta": "ta-IN",
    "gu": "gu-IN",
    "kn": "kn-IN",
    "ml": "ml-IN",
    "pa": "pa-IN",
    "or": "od-IN",
    "ur": "hi-IN",
    "as": "bn-IN",
}

DISCLAIMER_EN = (
    "This is general legal information, not a substitute for advice from a qualified lawyer."
)

SYSTEM_PROMPT = TRIAGE_SYSTEM_PROMPT

_triage_service = TriageService(system_prompt=SYSTEM_PROMPT, top_k=12)
_TRANSLATE_CHUNK_LIMIT = 500  # Sarvam Mayura works best with shorter text


def bcp47_target(lang: str) -> str:
    return UI_TO_BCP47.get(lang, "en-IN")


def _normalize_lang_code(code: str) -> str:
    return (code or "").strip()


def _translation_looks_unchanged(src: str, out: str) -> bool:
    return src.strip() == out.strip()


def _translate_with_fallback(text: str, *, source: str, target: str) -> str:
    """Translate once, then retry with base language codes if output is unchanged."""
    src = _normalize_lang_code(source)
    tgt = _normalize_lang_code(target)

    result = translate_text(text, source_language_code=src, target_language_code=tgt)
    if not _translation_looks_unchanged(text, result):
        return result

    src_base = src.split("-", 1)[0] if "-" in src else src
    tgt_base = tgt.split("-", 1)[0] if "-" in tgt else tgt

    # Retry only if the normalized call would actually differ.
    if src_base != src or tgt_base != tgt:
        retry = translate_text(text, source_language_code=src_base, target_language_code=tgt_base)
        if not _translation_looks_unchanged(text, retry):
            logger.info("Translate fallback succeeded with %s -> %s", src_base, tgt_base)
            return retry

    return result


def _chunked_translate(text: str, *, source: str, target: str) -> str:
    """Translate long text by splitting into paragraph-sized chunks."""
    paragraphs = text.split("\n")
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 1 > _TRANSLATE_CHUNK_LIMIT and current:
            chunks.append(current)
            current = para
        else:
            current = f"{current}\n{para}" if current else para
    if current:
        chunks.append(current)

    translated_parts = []
    for chunk in chunks:
        if not chunk.strip():
            translated_parts.append(chunk)
            continue
        try:
            result = _translate_with_fallback(chunk, source=source, target=target)
            translated_parts.append(result)
        except Exception as exc:
            logger.warning("Mayura chunk translate failed, keeping original: %s", exc)
            translated_parts.append(chunk)
    return "\n".join(translated_parts)


def _maybe_translate(text: str, *, source: str, target: str) -> str:
    if source == target:
        return text
    if not sarvam_configured():
        return text
    if len(text) > _TRANSLATE_CHUNK_LIMIT:
        return _chunked_translate(text, source=source, target=target)
    try:
        return _translate_with_fallback(text, source=source, target=target)
    except Exception as exc:
        logger.warning("Mayura translate failed, using original: %s", exc)
        return text


def text_to_query_english(user_text: str, lang: str) -> str:
    """Non-English typed input -> English for embedding/RAG (Mayura)."""
    t = user_text.strip()
    if not t:
        return t
    if lang == "en":
        return t
    if not sarvam_configured():
        logger.warning("SARVAM_API_KEY missing - using raw text for retrieval (degraded).")
        return t
    return _maybe_translate(t, source="auto", target="en-IN")


def resolve_user_message(text: str, audio_bytes: bytes | None, lang: str) -> tuple[str, str]:
    """Returns (user_bubble_text, query_english)."""
    text = (text or "").strip()

    # Prefer typed text over audio so chat input always wins.
    if text:
        q_en = text_to_query_english(text, lang)
        return (text, q_en)

    if audio_bytes:
        if not sarvam_configured():
            raise RuntimeError("Set SARVAM_API_KEY for voice input (Sarvam STT).")
        mode = os.environ.get("SARVAM_STT_MODE", "translate").strip()
        lang_hint = bcp47_target(lang) if mode == "transcribe" else None
        stt = speech_to_text_file(
            audio_bytes,
            filename="voice.wav",
            mode=mode,
            language_code=lang_hint,
        )
        tr = transcript_from_stt_response(stt)
        if mode == "translate":
            return (f"🎤 {tr}", tr.strip())
        q_en = _maybe_translate(tr, source="auto", target="en-IN")
        return (f"🎤 {tr}", q_en.strip())

    raise ValueError("Type a question or record audio before sending.")


def build_reply_markdown(assistant_en: str, cites: str, lang: str) -> str:
    """Build response markdown in session language with references and disclaimer."""
    sources_block = f"**References used**\n{cites}"

    if lang == "en" or not sarvam_configured():
        return (
            f"{assistant_en}\n\n---\n{sources_block}"
            f"\n\n---\n*{DISCLAIMER_EN}*"
        )

    tgt = bcp47_target(lang)
    body_translated = _maybe_translate(assistant_en, source="en-IN", target=tgt)
    disc_translated = _maybe_translate(DISCLAIMER_EN, source="en-IN", target=tgt)

    lang_label = dict(SARVAM_LANGUAGES).get(lang, lang)
    return (
        f"**{lang_label}:**\n\n{body_translated}\n\n"
        f"---\n{sources_block}"
        f"\n\n---\n*{disc_translated}*"
    )


def maybe_tts_bytes(text_markdown: str, lang: str, enabled: bool) -> bytes | None:
    if not enabled or not sarvam_configured():
        return None
    narrative = text_markdown.split("\n---\n", 1)[0]

    import re

    narrative = re.sub(r"^\*\*[^*]+:\*\*\s*", "", narrative.strip())
    plain = strip_markdown_for_tts(narrative)
    if not plain.strip():
        return None
    tgt = bcp47_target(lang)
    try:
        return text_to_speech_wav_bytes(plain, target_language_code=tgt)
    except Exception as exc:
        logger.warning("TTS failed: %s", exc)
        return None


def _load_query_logs() -> pd.DataFrame:
    """Load query logs from Delta or CSV fallback."""
    try:
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.getOrCreate()
        table = os.environ.get("NYAYA_LOG_TABLE", "workspace.default.query_logs")
        return spark.table(table).toPandas()
    except Exception:
        pass
    csv_path = os.environ.get("NYAYA_LOG_CSV", "/tmp/nyaya_query_logs.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return pd.DataFrame()


def _refresh_analytics() -> tuple[str, pd.DataFrame | None]:
    """Generate analytics summary markdown + dataframe preview."""
    import json as _json

    df = _load_query_logs()
    if df.empty:
        return "**No queries logged yet.** Start chatting to see analytics here!", None

    total = len(df)
    lines = ["## Analytics Dashboard\n", f"**Total queries:** {total}\n"]

    if "domain_detected" in df.columns:
        domain_counts = df["domain_detected"].value_counts()
        lines.append("### Legal Domain Breakdown")
        for domain, count in domain_counts.items():
            pct = count / total * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            lines.append(f"- **{domain}**: {count} ({pct:.0f}%) `{bar}`")
        lines.append("")

    if "user_lang" in df.columns:
        lang_counts = df["user_lang"].value_counts()
        lang_names = dict(SARVAM_LANGUAGES)
        lines.append("### Language Distribution")
        for lang_code, count in lang_counts.items():
            name = lang_names.get(lang_code, lang_code)
            lines.append(f"- **{name}** ({lang_code}): {count}")
        lines.append("")

    if "response_time_ms" in df.columns:
        avg_ms = df["response_time_ms"].mean()
        lines.append(f"### Performance\n- **Avg response time:** {avg_ms:.0f}ms")
        lines.append("")

    if "sections_cited" in df.columns:
        all_sections = []
        for raw in df["sections_cited"].dropna():
            try:
                secs = _json.loads(raw) if isinstance(raw, str) else raw
                if isinstance(secs, list):
                    all_sections.extend(secs)
            except Exception:
                pass
        if all_sections:
            from collections import Counter

            sec_counts = Counter(all_sections).most_common(10)
            lines.append("### Most Cited Legal Provisions (Top 10)")
            for sec, cnt in sec_counts:
                lines.append(f"- **{sec}**: cited {cnt} time(s)")
            lines.append("")

    summary_md = "\n".join(lines)
    display_cols = [
        c
        for c in ["timestamp", "user_lang", "domain_detected", "query_text", "response_time_ms"]
        if c in df.columns
    ]
    recent = df[display_cols].tail(20).iloc[::-1] if display_cols else df.tail(20).iloc[::-1]

    return summary_md, recent


def _init_session_state() -> None:
    st.session_state.setdefault("lang", "en")
    st.session_state.setdefault("tts_on", True)
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("history_pairs", [])
    st.session_state.setdefault("latest_tts", None)
    st.session_state.setdefault("analytics_summary", "")
    st.session_state.setdefault("analytics_table", None)


def _apply_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700&family=Playfair+Display:wght@700&display=swap');

        :root {
            --bg: #f7f6f2;
            --paper: #ffffff;
            --ink: #24303a;
            --muted: #60707d;
            --brand: #2f6f7a;
            --brand-strong: #245760;
            --border: #dbe3ea;
            --chat-user: #eef6f8;
            --chat-assistant: #ffffff;
            --focus: #2f6f7a;
        }

        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"] {
            font-family: 'Manrope', 'Segoe UI', sans-serif;
            color: var(--ink);
            background:
                radial-gradient(920px 460px at 10% -10%, #e8f1f4 0%, transparent 44%),
                radial-gradient(760px 420px at 100% -6%, #eef2f7 0%, transparent 40%),
                var(--bg);
        }

        .stApp,
        .stApp p,
        .stApp span,
        .stApp label,
        .stApp li,
        .stMarkdown,
        .stCaptionContainer {
            color: var(--ink);
        }

        h1, h2, h3, h4 {
            font-family: 'Playfair Display', Georgia, serif;
            color: #2b2014;
        }

        [data-testid="stSidebar"] {
            background: var(--paper);
            border-left: 1px solid var(--border);
        }

        [data-testid="stSidebarContent"] {
            background: linear-gradient(180deg, #ffffff 0%, #f8fbfc 100%);
        }

        [data-testid="stSidebar"] * {
            color: var(--ink);
        }

        [data-baseweb="select"] > div,
        [data-baseweb="input"] > div,
        [data-baseweb="textarea"] > div,
        .stTextInput input,
        .stTextArea textarea,
        .stSelectbox div[data-baseweb="select"] > div,
        .stMultiSelect div[data-baseweb="select"] > div,
        .stFileUploader [data-testid="stFileUploaderDropzone"],
        [data-testid="stChatInput"] textarea,
        [data-testid="stChatInput"] input {
            background: #ffffff !important;
            color: var(--ink) !important;
            border-color: var(--border) !important;
        }

        [data-baseweb="select"] > div,
        [data-baseweb="input"] > div,
        [data-baseweb="textarea"] > div {
            box-shadow: none !important;
        }

        [data-baseweb="select"] [role="option"],
        [data-baseweb="popover"] * {
            color: var(--ink) !important;
        }

        [data-baseweb="popover"],
        [data-baseweb="menu"],
        [role="listbox"] {
            background: #ffffff !important;
            border: 1px solid var(--border) !important;
        }

        [role="option"] {
            background: #ffffff !important;
        }

        [role="option"][aria-selected="true"],
        [role="option"]:hover {
            background: #eef6f8 !important;
        }

        [data-testid="stChatInput"] {
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 2px;
        }

        [data-testid="stChatMessage"] {
            border: 1px solid var(--border);
            border-radius: 14px;
            background: var(--chat-assistant);
            padding: 0.3rem 0.7rem;
            box-shadow: 0 4px 16px rgba(24, 43, 64, 0.04);
        }

        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
            background: var(--chat-user);
        }

        .stButton > button,
        .stDownloadButton > button,
        [data-testid="stBaseButton-primary"] {
            border: none;
            color: #ffffff !important;
            background: linear-gradient(135deg, var(--brand), var(--brand-strong));
            box-shadow: 0 6px 16px rgba(36, 87, 96, 0.24);
        }

        .stButton > button *,
        .stDownloadButton > button * {
            color: #ffffff !important;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover,
        [data-testid="stBaseButton-primary"]:hover {
            filter: brightness(1.03);
            transform: translateY(-1px);
        }

        .stButton > button:focus, .stDownloadButton > button:focus,
        [data-testid="stChatInput"] textarea:focus,
        [data-testid="stChatInput"] input:focus {
            outline: 2px solid var(--focus) !important;
            outline-offset: 1px;
        }

        [data-testid="stDataFrame"],
        [data-testid="stTable"] {
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: 10px;
        }

        [data-testid="stToolbar"],
        [data-testid="stDecoration"] {
            background: transparent !important;
        }

        .subtle-note {
            color: var(--muted);
            font-size: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _submit_turn_with_feedback(
    *,
    text: str,
    audio_bytes: bytes | None,
    user_preview: str | None = None,
    query_text_for_log: str | None = None,
) -> None:
    preview = user_preview if user_preview is not None else (text.strip() if text.strip() else "🎤 Processing voice input...")
    st.session_state.messages.append({"role": "user", "content": preview})

    with st.chat_message("user"):
        st.markdown(preview)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        with st.spinner("Preparing legal guidance..."):
            try:
                user_show, reply_md, assistant_en, domain_str, elapsed_ms, q_en = _generate_reply(
                    text=text,
                    audio_bytes=audio_bytes,
                )

                st.session_state.messages[-1]["content"] = user_show
                st.session_state.history_pairs.append([user_show, reply_md])
                st.session_state.messages.append({"role": "assistant", "content": reply_md})
                st.session_state.latest_tts = maybe_tts_bytes(
                    reply_md,
                    st.session_state.lang,
                    st.session_state.tts_on,
                )

                try:
                    log_query(
                        user_lang=st.session_state.lang,
                        query_text=query_text_for_log if query_text_for_log is not None else (text or "(audio)"),
                        query_en=q_en,
                        domain_detected=domain_str,
                        response_en=assistant_en,
                        response_time_ms=elapsed_ms,
                    )
                except Exception:
                    logger.debug("Query logging failed (non-fatal)", exc_info=True)

                response_placeholder.markdown(reply_md)
            except Exception as exc:
                err = f"**Error:** {exc}"
                st.session_state.messages.append({"role": "assistant", "content": err})
                response_placeholder.markdown(err)


def _generate_reply(*, text: str, audio_bytes: bytes | None) -> tuple[str, str, str, str, int, str]:
    lang = st.session_state.lang
    history = st.session_state.history_pairs

    user_show, q_en = resolve_user_message(text, audio_bytes, lang)
    assistant_en, cites, domain_str, elapsed_ms = _triage_service.answer(q_en, history)
    reply_md = build_reply_markdown(assistant_en, cites, lang)
    return (user_show, reply_md, assistant_en, domain_str, elapsed_ms, q_en)


def render_app() -> None:
    st.set_page_config(
        page_title="Nyaya-Sahayak",
        page_icon="⚖️",
        layout="centered",
        initial_sidebar_state="expanded",
    )
    _apply_styles()
    _init_session_state()

    st.title("Nyaya-Sahayak · न्याय सहायक")
    st.caption("Your legal first-response assistant powered by Databricks and Sarvam AI.")

    pending_text: str | None = None
    pending_audio_bytes: bytes | None = None
    pending_user_preview: str | None = None
    pending_query_text_for_log: str | None = None

    with st.sidebar:
        st.subheader("Session")
        lang_map = {label: code for code, label in SARVAM_LANGUAGES}
        lang_codes = list(lang_map.values())
        lang_index = lang_codes.index(st.session_state.lang) if st.session_state.lang in lang_codes else 0
        selected_label = st.selectbox(
            "Language",
            options=list(lang_map.keys()),
            index=lang_index,
            help="Questions are retrieved in English and answers are returned in your selected language.",
        )
        st.session_state.lang = lang_map[selected_label]

        if st.session_state.lang != "en" and not sarvam_configured():
            st.warning("Translation is unavailable: SARVAM_API_KEY is missing in app runtime.")

        st.session_state.tts_on = st.toggle("Read answer aloud", value=st.session_state.tts_on)

        topic = st.selectbox("Quick start prompt", options=["Choose a topic..."] + list(TOPIC_SEEDS.keys()))
        if st.button("Ask quick prompt", use_container_width=True) and topic in TOPIC_SEEDS:
            pending_text = TOPIC_SEEDS[topic]
            pending_audio_bytes = None
            pending_user_preview = pending_text
            pending_query_text_for_log = pending_text

        st.markdown("---")
        st.subheader("Voice")
        audio_file = None
        if hasattr(st, "audio_input"):
            audio_file = st.audio_input("Record your question")
        else:
            st.info("Audio recording is not available in this Streamlit version. Upload a WAV file instead.")
            audio_file = st.file_uploader("Upload WAV", type=["wav"])
        if st.button("Send voice", use_container_width=True):
            if audio_file is None:
                st.warning("Record audio first.")
            else:
                pending_text = ""
                pending_audio_bytes = audio_file.getvalue()
                pending_user_preview = "🎤 Processing voice input..."
                pending_query_text_for_log = "(audio)"

        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.history_pairs = []
            st.session_state.latest_tts = None
            st.rerun()

        with st.expander("Usage analytics", expanded=False):
            if st.button("Refresh analytics", use_container_width=True):
                summary, table = _refresh_analytics()
                st.session_state.analytics_summary = summary
                st.session_state.analytics_table = table

            if st.session_state.analytics_summary:
                st.markdown(st.session_state.analytics_summary)
            else:
                st.markdown("**Click 'Refresh analytics' to load usage data.**")

            if st.session_state.analytics_table is not None:
                st.dataframe(st.session_state.analytics_table, use_container_width=True, hide_index=True)

        st.markdown(
            "<p class='subtle-note'>General legal information only. For your specific situation, consult a qualified lawyer.</p>",
            unsafe_allow_html=True,
        )

    st.markdown(f"**Session language:** {selected_label}")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if pending_text is not None or pending_audio_bytes is not None:
        _submit_turn_with_feedback(
            text=pending_text or "",
            audio_bytes=pending_audio_bytes,
            user_preview=pending_user_preview,
            query_text_for_log=pending_query_text_for_log,
        )

    user_text = st.chat_input("Ask your legal question in any supported language...")
    if user_text:
        _submit_turn_with_feedback(
            text=user_text,
            audio_bytes=None,
            user_preview=user_text,
            query_text_for_log=user_text,
        )

    if st.session_state.latest_tts:
        st.audio(st.session_state.latest_tts, format="audio/wav")

    st.markdown(
        "<p class='subtle-note'>⚖️ This is informational guidance only - not a substitute for legal counsel. "
        "Consult a qualified lawyer for your specific situation.<br>"
        "Powered by Databricks (Llama Maverick + Vector Search) and Sarvam AI "
        "(translation, speech-to-text, text-to-speech)</p>",
        unsafe_allow_html=True,
    )


def _load_secrets_from_scope() -> None:
    """Load secrets from Databricks secret scope into env vars (for Databricks Apps)."""
    mapping = {
        "SARVAM_API_KEY": ("nyaya-dhwani", "sarvam_api_key"),
    }
    for env_var, (scope, key) in mapping.items():
        if os.environ.get(env_var, "").strip():
            continue
        try:
            from databricks.sdk import WorkspaceClient

            w = WorkspaceClient()
            val = w.secrets.get_secret(scope=scope, key=key)
            if val and val.value:
                import base64

                try:
                    decoded = base64.b64decode(val.value).decode("utf-8")
                except Exception:
                    decoded = val.value
                os.environ[env_var] = decoded
                logger.info("Loaded %s from secret scope %s/%s", env_var, scope, key)
        except Exception as exc:
            logger.warning("Could not load %s from secret scope: %s", env_var, exc)


def _is_streamlit_runtime() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


def _bootstrap_streamlit() -> None:
    script_path = str(Path(__file__).resolve())
    port = os.environ.get("DATABRICKS_APP_PORT") or os.environ.get("PORT") or "8000"
    cmd = [
        "streamlit",
        "run",
        script_path,
        "--server.port",
        str(port),
        "--server.address",
        "0.0.0.0",
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    logger.info("Starting Streamlit via CLI on port %s", port)
    os.execvp(cmd[0], cmd)


def main() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    _load_secrets_from_scope()

    if _is_streamlit_runtime():
        render_app()
        return

    _bootstrap_streamlit()


if __name__ == "__main__":
    main()
