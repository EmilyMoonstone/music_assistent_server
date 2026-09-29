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
# the jingle picked for a clip when its script was written, "" for none
ATTR_JINGLE = "ai_radio_jingle"
# per-section opt-in: a break from a section that allows it may be split so its tail
# carries over the next record's intro (a "post")
ATTR_ALLOW_POST = "ai_radio_allow_post"

# A post is the tail of one continuous break mixed over the next record's intro. With a
# break of B seconds and W seconds of intro before the vocal:
#
#   overlap = min(W - POST_TAIL_GAP, B - POST_MIN_HEAD_SECONDS)
#
# the break airs alone for B - overlap seconds, then the record starts underneath it and
# the same recording's last `overlap` seconds play over the intro.
POST_TAIL_GAP = 0.4  # seconds of music between the end of the voice and the vocal entry
# what a host can change about its posts: the gap before the vocal, the longest stretch it
# talks over the intro (0 for no cap), and how far the music drops under the voice
POST_GAP_RANGE = (0.0, 3.0)
POST_MAX_RANGE = (0.0, 30.0)
DEFAULT_POST_DUCK_PERCENT = 60
POST_DUCK_RANGE = (0, 90)
POST_MIN_SECONDS = 1.5  # shortest overlap worth doing; below it the break plays whole
POST_MIN_HEAD_SECONDS = 1.0  # the break keeps at least this much for its own queue item
# MA's lyrics lookup walks every metadata provider; past this budget the break plays whole
POST_LYRICS_TIMEOUT = 8.0
# a postable break is rendered once into a local, levelled copy; a render this slow is wedged
POST_STAGE_TIMEOUT = 20
# staged copies: their file name prefix, and the age past which one is a leftover to delete
POST_CLIP_PREFIX = "ma_ai_radio_post_"
POST_CLIP_MAX_AGE = 3600
# the staged copy is the clip's PCM wrapped in WAV, so ffmpeg reads it without format hints
POST_STAGED_FORMAT = AudioFormat(
    content_type=ContentType.WAV,
    sample_rate=TTS_CLIP_PCM_FORMAT.sample_rate,
    bit_depth=TTS_CLIP_PCM_FORMAT.bit_depth,
    channels=TTS_CLIP_PCM_FORMAT.channels,
)
# the jingle picked to close a clip, leading into the next song, "" for none
ATTR_JINGLE_AFTER = "ai_radio_jingle_after"
# what the clip's section says about a jingle ahead of and after its break, see
# JINGLE_SLOT_MODES
ATTR_JINGLE_BEFORE_MODE = "ai_radio_jingle_before_mode"
ATTR_JINGLE_AFTER_MODE = "ai_radio_jingle_after_mode"

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

# a host can dress its breaks with sound: jingles from its own library ahead of them, and a
# music bed under everything it says. A jingle is either the gong Music Assistant ships, or
# like a bed a file path or URL the user supplies
EFFECT_BUILTIN_JINGLE = "builtin"
# the occasions a jingle can be tagged for; a jingle tagged for none of them is general
JINGLE_OCCASION_TAGS = ("general", "news", "weather", "intro", "outro")
# the time-of-day tags, each with the local hours [start, end) it covers
JINGLE_TIME_TAGS: dict[str, tuple[int, int]] = {
    "morning": (5, 10),
    "daytime": (10, 17),
    "evening": (17, 22),
    "late_night": (22, 5),
}
# the slot each show jingle occasion airs in
JINGLE_SLOT_OCCASIONS = {"start_of_playlist": "intro", "end_of_playlist": "outro"}
# how often a plain transition opens with a jingle, in percent
DEFAULT_JINGLE_CHANCE = 20
JINGLE_SELECTION_MODES = ("ai", "random")
DEFAULT_JINGLE_SELECTION = "ai"
MAX_JINGLES = 50
# jingles are picked from Home Assistant's media folder, the one folder every install
# shares with Music Assistant; browsing stops at its edge
JINGLE_MEDIA_ROOT = "/media"
# announcements only play http URLs, so a jingle is previewed through a short-lived link
# on the stream server that serves just that one file
JINGLE_PREVIEW_ROUTE = "/ai_radio/jingle_preview"
JINGLE_PREVIEW_SECONDS = 120
JINGLE_FILE_EXTENSIONS = frozenset(
    {".mp3", ".m4a", ".aac", ".flac", ".ogg", ".oga", ".opus", ".wav", ".aiff", ".aif", ".wma"}
)
MAX_JINGLE_TEXT_CHARS = 300
# a section's say on the jingle ahead of and after its break: "auto" leaves it to the host
# (and, for the one after, to the LLM), "always" and "never" settle it
JINGLE_SLOT_MODES = ("auto", "always", "never")
DEFAULT_JINGLE_SLOT_MODE = "auto"
# a break closes with a jingle of its own accord at most this often, in minutes per host
DEFAULT_JINGLE_AFTER_GAP_MINUTES = 30
JINGLE_AFTER_GAP_RANGE = (0, 240)
# a song singing this soon after it starts leaves no intro to talk over, so a jingle
# bridges into it instead
JINGLE_AFTER_EARLY_VOCAL_SECONDS = 3.0
# the AI listens to a jingle once when it is filed, and suggests how to tag it
JINGLE_ANALYSIS_PROMPT = (
    "Listen to the attached radio jingle and file it for a radio automation. Reply with "
    'JSON only, no prose: {{"tags": [], "text": "", "style": ""}}. '
    "tags: any of {occasions} the jingle suits (general for a plain station ident), any of "
    "{times} it suits by its mood, plus up to {style_tags} short lowercase English style "
    "tags for genre, mood and energy, like indie, calm or upbeat. "
    "text: the words it sings or speaks, in their own language, empty if there are none. "
    "style: one short sentence on its sound and mood, in the language of the locale "
    "'{language}'."
)
JINGLE_ANALYSIS_STYLE_TAGS = 4
MAX_JINGLE_STYLE_CHARS = 200
# what a jingle says is cut to this in the prompt: enough to pick by, few tokens
JINGLE_PROMPT_TEXT_CHARS = 120
# the LLM that writes a break also picks its jingles, answering on lines of their own
JINGLE_LIST_HEADER = "Jingles (number. [tags] words):"
JINGLE_BEFORE_INSTRUCTION = (
    "Start your reply with a line 'JINGLE: <number>' for the jingle right before you speak "
    "({numbers})."
)
JINGLE_AFTER_AUTO_INSTRUCTION = (
    "Then a line 'AFTER: <number or none>' for a jingle after you, into the next song "
    "({numbers}). Mostly none: only where it fits, like closing the news or when the song "
    "sings right away.{onset}"
)
JINGLE_AFTER_ALWAYS_INSTRUCTION = (
    "Then a line 'AFTER: <number>' for the jingle after you, into the next song ({numbers})."
)
JINGLE_AFTER_ONSET_HINT = " The next song's vocals start after {seconds:.0f}s."
JINGLE_CHOICE_CLOSING = (
    "Pick by time of day, the music around and the words. The listeners hear the jingles "
    "themselves, so never say, quote or paraphrase their words. Then the script."
)
# a jingle phrase this short is too common to take out of a script, like "erste Platte"
MIN_ECHOED_PHRASE_WORDS = 3
# how the song before blends into a break: a hard cut, a crossfade, or a talk-up where the
# break starts at full level over the song's fading outro
LEAD_IN_MODES = ("cut", "crossfade", "talk_up")
DEFAULT_LEAD_IN = "cut"
DEFAULT_LEAD_IN_SECONDS = 3
LEAD_IN_SECONDS_RANGE = (1, 8)
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

# how a show orders its songs: shuffled, in the playlist's own order, or as a running order
# the AI puts together like a music director would
TRACK_ORDER_SHUFFLE = "shuffle"
TRACK_ORDER_PLAYLIST = "playlist"
TRACK_ORDER_AI = "ai"
TRACK_ORDER_MODES = (TRACK_ORDER_SHUFFLE, TRACK_ORDER_PLAYLIST, TRACK_ORDER_AI)
# the AI orders a random pick of this many songs, which keeps the prompt, and the wait
# before the show starts, bounded for playlists of any size
DEFAULT_AI_ORDER_MAX_TRACKS = 100
AI_ORDER_MAX_TRACKS_RANGE = (5, 500)
DEFAULT_AI_ORDER_PROMPT = (
    "You are the music director of this station and put together the running order of "
    "the show. The show starts at <timestamp>: open with songs that suit that time of day, "
    "and let the energy follow how the hours go. Keep it varied: never play the same artist "
    "twice in a row, and change genre, era and tempo now and then. Still let neighbouring "
    "songs belong together, so every transition feels intended through a shared mood, "
    "tempo, sound or story. Build small arcs instead of a random walk."
)
MAX_LISTENER_WISH_CHARS = 500
# appended to every running-order request, custom prompts included, so the reply parses
AI_ORDER_REPLY_INSTRUCTION = (
    "Reply with a JSON array of the song numbers in the order they should play, every "
    "number exactly once, and nothing else."
)

# HA drops a tts_proxy token 60s after its last use at the lowest configurable time_memory
CLIP_STREAMDETAILS_EXPIRATION = 60

# a TTS engine that hands out a URL (Home Assistant's tts_proxy) is read once into a local
# copy: the link dies a minute after its last use, and some setups close the transfer with an
# error after the last byte, which would otherwise abort the clip on air. The copy is plain PCM
# in a WAV file, in the format the clip is played at
CLIP_COPY_PREFIX = "ma_ai_radio_clip_"
CLIP_COPY_FORMAT = AudioFormat(
    content_type=ContentType.WAV,
    sample_rate=TTS_CLIP_PCM_FORMAT.sample_rate,
    bit_depth=TTS_CLIP_PCM_FORMAT.bit_depth,
    channels=TTS_CLIP_PCM_FORMAT.channels,
)
# how long a local copy is kept and handed out again, and when a leftover one is deleted
CLIP_COPY_LIFETIME = 3600
CLIP_FETCH_TIMEOUT = 60
# a copy shorter than this holds no speech, it is a failed fetch (0.2s of 48 kHz stereo s16)
MIN_CLIP_COPY_BYTES = 44 + 38400

# a cached clip with less life than this left is not worth handing out, so it is re-minted
MIN_CLIP_MEDIA_LIFETIME = 5
