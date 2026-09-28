"""Constants for the AI Radio plugin."""

from __future__ import annotations

from typing import Any

from music_assistant_models.enums import ContentType
from music_assistant_models.media_items import AudioFormat

CONF_AI_ENGINE = "ai_engine"
CONF_BREAK_MEMORY = "break_memory"
CONF_TTS_ENGINE = "tts_engine"
CONF_TTS_LOUDNESS_BOOST = "tts_loudness_boost"
CONF_TIMEZONE = "timezone"
CONF_WEATHER_CITY = "weather_city"
CONF_WEATHER_COUNTRY = "weather_country"
CONF_WEATHER_PROVIDER = "weather_provider"
CONF_WEATHER_TIMEOUT = "weather_timeout_seconds"

# providers load concurrently, so the plugin supplying the engines may still be
# loading when AI Radio initializes: wait this long for it before giving up
ENGINE_DISCOVERY_TIMEOUT = 30

# grace period for an engine that disappears while AI Radio is loaded. Generous enough
# to sit out a Home Assistant restart, so a running show is not torn down for it
ENGINE_RECHECK_GRACE = 300

# how long to wait before reloading after an engine stayed missing, matching the
# cadence the load path uses for its own retries
ENGINE_RETRY_DELAY = 120

TRANSLATION_OWNER = "provider.ai_radio"

DEFAULT_LLM_INSTRUCTIONS = (
    "Host personality: warm, sharp, music-literate, and slightly premium "
    "without sounding formal. Program instructions: write for spoken delivery, "
    "keep segments concise, avoid bullet-point phrasing, avoid clichés, "
    "mention concrete details when available, and maintain a believable "
    "radio flow between sections."
)
# appended to every AI query on top of the station's own instructions: how a name has to be
# spelled to survive the TTS engine is a pipeline concern, not a per-station style choice
TTS_PRONUNCIATION_INSTRUCTIONS = (
    "The output is sent directly to a text-to-speech engine. "
    "Always write names exactly as they should be spoken aloud. Replace stylized spellings, "
    "acronyms, abbreviations, and unusual artist or band names with their natural spoken "
    "equivalents. Never include the original spelling, pronunciation explanation, phonetic "
    "notation, or both versions. Output only the spoken version. Examples: INXS → In Excess; "
    "Mi-Sex → My Sex; P!nk → Pink; blink-182 → Blink One Eighty-Two. If a name could be "
    "mispronounced by the TTS engine, rewrite it into the clearest natural spoken form "
    "without explaining the change. "
    "Names and titles often stay in their original language while the voice reads everything "
    "with the pronunciation rules of the script's language. When a name would be mangled that "
    "way, respell it phonetically for the script's language so it still sounds like the "
    "original; leave names that already read correctly untouched."
)
MERGE_SECTION_PROMPT = (
    "Merge the drafts below into one coherent radio break. "
    "Preserve factual content, remove duplication, and make the "
    "final segment sound like one host speaking naturally.\n"
    "<section_drafts>"
)
DEFAULT_WEATHER_PROVIDER = "open_meteo"
DEFAULT_WEATHER_TIMEOUT_SECONDS = 20

# countries and US territories that use Fahrenheit for everyday temperatures
FAHRENHEIT_COUNTRY_CODES = frozenset(
    {"US", "PR", "GU", "VI", "AS", "MP", "LR", "MM", "BS", "BZ", "KY", "PW"}
)
DEFAULT_MAX_CONCURRENT_RUNS = 1
MAX_FINISHED_SESSIONS = 20

# a show whose playback never starts within this window is declared failed
SHOW_START_TIMEOUT_SECONDS = 300

# last-resort guard so a wedged engine fails the clip instead of hanging the session.
# Kept above the deadlines the engines apply themselves (120s in the OpenAI-compatible
# providers), so their own, more specific error is the one that surfaces.
AI_QUERY_TIMEOUT_SECONDS = 180

# ffprobe reports no status code, so its message is all we have to spot a failed render
TTS_SERVER_ERROR_MARKERS = ("Server returned 5XX", "HTTP error 5")

DEFAULT_TTS_LOUDNESS_BOOST = 3

# speech carries ~16 dB between its average level and its peaks, so a plain gain that
# reaches the target clips instead. speechnorm evens the clip out so the level is carried
# by the whole clip, the trim then places it, and the limiter backstops the peaks
TTS_SPEECHNORM_FILTER = "speechnorm=e=12.5:r=0.0005:l=1"
TTS_PEAK_CEILING_DB = -1.5

# one measurement stands in for every clip an engine voices, but a fragment of a few
# words is not representative enough of its level to become that reference
MIN_LOUDNESS_REFERENCE_SECONDS = 2

# a clip is seconds of audio, so a measurement that takes this long is a wedged fetch
LOUDNESS_MEASURE_TIMEOUT = 60

# spoken clips are handed to MA already decoded, so the filter chain runs once here
# instead of once per output
TTS_CLIP_PCM_FORMAT = AudioFormat(
    content_type=ContentType.PCM_S16LE,
    sample_rate=48000,
    bit_depth=16,
    channels=2,
)

SUPPORTED_FEATURES: set[Any] = set()
EMPTY_SECTION_ID = "EMPTY_SECTION"
VALID_WEB_SEARCH_MODES = {"disabled", "allow", "force"}
WEB_SEARCH_MODE_RANK = {"disabled": 0, "allow": 1, "force": 2}

# QueueItem.extra_attributes keys carrying a clip's pending render state. Scalars only —
# extra_attributes is serialized to clients and persisted with the queue.
ATTR_SESSION_ID = "ai_radio_session_id"
ATTR_STATION_ID = "ai_radio_station_id"
ATTR_PROMPT = "ai_radio_prompt"
ATTR_MAX_CHARS = "ai_radio_max_chars"
ATTR_WEB_SEARCH_MODE = "ai_radio_web_search_mode"
ATTR_RENDERED_TEXT = "ai_radio_rendered_text"
ATTR_HOST_ID = "ai_radio_host_id"
ATTR_QUEUE_DJ = "ai_radio_queue_dj"
ATTR_GAP_NEXT_ID = "ai_radio_gap_next_id"
ATTR_WEATHER_REQUIRED = "ai_radio_weather_required"
ATTR_SLOT_WHEN = "ai_radio_slot_when"

# placeholders resolved at render time rather than at plan time, so the aired script
# reflects the moment it plays
DEFERRED_PLACEHOLDERS = frozenset({"<timestamp>", "<weather_hourly>", "<weather_daily>"})

# the deferred placeholders that need a successful weather fetch to say anything at all
WEATHER_PLACEHOLDER_TOKENS = ("<weather_hourly>", "<weather_daily>")

# substituted for an unresolved weather token in clips that still air
NO_WEATHER_DATA_INSTRUCTION = (
    "(no weather data available - leave out all weather talk, do not invent a forecast)"
)

# every break is written fresh by the LLM, which on its own drifts back to the same obvious
# angle each time. The breaks a host aired are remembered and fed back into its later prompts
DEFAULT_BREAK_MEMORY = True
RECENT_BREAKS_PLACEHOLDER = "<recent_breaks>"
RECENT_NEWS_PLACEHOLDER = "<recent_news>"
BREAK_MEMORY_PLACEHOLDERS = (RECENT_BREAKS_PLACEHOLDER, RECENT_NEWS_PLACEHOLDER)
# how many of a host's latest breaks a prompt sees, and how far back its news reaches
RECENT_BREAKS_IN_PROMPT = 8
RECENT_NEWS_WINDOW_HOURS = 12
# what is kept per host: enough to cover the news window of a chatty host, while a break
# from days ago says nothing about what sounds repetitive now
BREAK_MEMORY_MAX_BREAKS = 30
BREAK_MEMORY_MAX_NEWS = 30
BREAK_MEMORY_RETENTION_HOURS = 48
# a remembered break only has to carry its topic and phrasing, not the whole script
BREAK_MEMORY_BREAK_CHARS = 350
BREAK_MEMORY_NEWS_CHARS = 700
BREAK_MEMORY_EMPTY = "(nothing yet)"
BREAK_MEMORY_BREAKS_INSTRUCTION = (
    "What you said in your most recent breaks on air, oldest first. Do not repeat "
    "their topics, angles, images, jokes, openings, sign-offs or signature phrases, and do "
    "not start the same way. Find a fresh angle:"
)
BREAK_MEMORY_NEWS_INSTRUCTION = (
    "News you already reported in the last hours, oldest first. Pick other stories. Only "
    "return to one of these when there is a genuinely new development, and then present it "
    "as an update:"
)

# a host can dress its breaks with sound: a jingle ahead of the news, one ahead of the intro
# and sign-off of a show, and a music bed under everything it says. A jingle is either the
# gong Music Assistant ships, or like a bed a file path or URL the user supplies
EFFECT_BUILTIN_JINGLE = "builtin"
EFFECT_SOURCE_KEYS = ("news_jingle", "show_jingle", "music_bed")
# slots a show jingle plays in, the moments a real station would play its ident
SHOW_JINGLE_SLOTS = frozenset({"start_of_playlist", "end_of_playlist"})
# how far below the voice the bed sits, in dB
DEFAULT_MUSIC_BED_LEVEL = -18
MUSIC_BED_LEVEL_RANGE = (-40, -6)
# the voice comes in this long before the jingle ends, so the two overlap like on air
JINGLE_VOICE_OVERLAP_SECONDS = 0.4
MUSIC_BED_FADE_IN_SECONDS = 1.0
# the bed plays on this long after the last word, fading out over it
MUSIC_BED_TAIL_SECONDS = 2.0
# only the start of a long bed is measured, its level does not change much after that
EFFECT_MEASURE_SECONDS = 60
# the level a sound is brought to when neither the queue nor the voice gives a reference
DEFAULT_EFFECT_LOUDNESS = -16.0

# HA drops a tts_proxy token 60s after its last use at the lowest configurable time_memory
CLIP_STREAMDETAILS_EXPIRATION = 60

# a cached clip with less life than this left is not worth handing out, so it is re-minted
MIN_CLIP_MEDIA_LIFETIME = 5
