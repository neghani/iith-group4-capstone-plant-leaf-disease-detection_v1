"""Demo-friendly stage logging with ANSI colors.

One loud, emoji-tagged, color-coded line per pipeline event so that, in a
split-screen demo (container logs on one side, the app on the other), every
stage is obvious at a glance — each layer has its own color:

    🌐 API        blue      (requests in, verdicts out)
    🔍 QUALITY    cyan      (gate pass/reject)
    🧠 AGENT      magenta   (LangGraph nodes + routing)
    📚 KNOWLEDGE  yellow    (KB lookups)
    📝 REPORT     green     (report assembly + LLM summary hand-off)
    💬 LLM        sky blue  (model calls)
    🛡️ SAFETY     green     (final guardrails)
    🗨️ CHAT       orange    (follow-up questions)

Symbols are colored by outcome: ▶ starts (cyan), ✔ success (green),
✖ failure (red), → decisions (yellow). Timings are dim.

Colors are on by default so they survive `docker compose logs -f` (raw
pass-through). Set `NO_COLOR=1` or `DEMO_LOG_COLOR=0` to disable (e.g. when
piping logs to a file).

Usage:
    from app.demo_log import demo, Stage

    demo("QUALITY", "✔", "PASS — brightness 0.62 · leaf 12.4%")
    with Stage("LLM") as st:
        ...
        st.ok("reply in 1.8 s (612 chars)")   # exit logs ✔/✖ with elapsed ms
"""

import logging
import os
import time

_log = logging.getLogger("leafcheck.demo")

# --- ANSI helpers -----------------------------------------------------------
_COLOR = os.environ.get("NO_COLOR") is None and os.environ.get("DEMO_LOG_COLOR", "1") != "0"

RESET = "\x1b[0m"
DIM = "\x1b[2m"
BOLD = "\x1b[1m"

# stage -> color code (distinct hue per layer of the pipeline)
STAGE_COLORS = {
    "API": "94",        # bright blue
    "QUALITY": "96",    # bright cyan
    "CROP": "36",       # cyan
    "AGENT": "95",      # bright magenta
    "KNOWLEDGE": "93",  # bright yellow
    "REPORT": "92",     # bright green
    "LLM": "94",        # bright blue
    "SAFETY": "92",     # bright green
    "CHAT": "33",       # orange/yellow
}

SYMBOL_COLORS = {
    "▶": "96",  # cyan — starting
    "✔": "92",  # green — success
    "✖": "91",  # red — failure
    "→": "93",  # yellow — decision/route
}


def _c(code: str, text: str) -> str:
    return f"\x1b[{code}m{text}{RESET}" if _COLOR else text


# icon + fixed-width tag keeps columns aligned across stages in the terminal
STAGES = {
    "API": "🌐",
    "QUALITY": "🔍",
    "CROP": "✂️",
    "AGENT": "🧠",
    "KNOWLEDGE": "📚",
    "REPORT": "📝",
    "LLM": "💬",
    "SAFETY": "🛡️",
    "CHAT": "🗨️",
}

_TAG_WIDTH = 9

if not _log.handlers:
    # The demo logger always owns its handler (propagate stays off below), so the
    # lines reach container stdout no matter which module is imported first.
    # The timestamp is dim; the rest of the line carries its own color codes.
    _handler = logging.StreamHandler()
    _stamp = f"{DIM}%(asctime)s{RESET} " if _COLOR else "%(asctime)s "
    _handler.setFormatter(logging.Formatter(_stamp + "%(message)s", datefmt="%H:%M:%S"))
    _log.addHandler(_handler)
_log.setLevel(logging.INFO)
_log.propagate = False  # one handler, one line per event — no duplicates


def _emit(stage: str, symbol: str, msg: str) -> None:
    icon = STAGES.get(stage, "🎬")
    tag = _c(BOLD + ";" + STAGE_COLORS.get(stage, "37"), stage.ljust(_TAG_WIDTH))
    sym = _c(SYMBOL_COLORS.get(symbol, "37"), symbol)
    _log.info("%s %s %s %s", icon, tag, sym, msg)


def demo(stage: str, symbol: str, msg: str) -> None:
    """Log one demo line: `demo("LLM", "✔", "summary rewritten in 1.8s")`."""
    _emit(stage, symbol, msg)


def banner(stage: str, msg: str) -> None:
    """Loud multi-line box for request start / result — the 'chapter heading' of a demo."""
    icon = STAGES.get(stage, "🎬")
    color = STAGE_COLORS.get(stage, "37")
    rule = _c(color, "─" * 70)
    _log.info("")
    _log.info("%s %s", icon, rule)
    _log.info("%s %s %s", icon, _c(BOLD + ";" + color, stage.ljust(_TAG_WIDTH)), msg)
    _log.info("%s %s", icon, rule)


class Stage:
    """Context manager that times a stage and logs ✔/✖ on exit unless you already logged.

    with Stage("QUALITY") as st:
        ...
        st.ok("PASS — brightness 0.62")   # logs elapsed time too
    """

    def __init__(self, stage: str, start_msg: str = ""):
        self.stage = stage
        self._done = False
        self._t0 = time.perf_counter()
        if start_msg:
            self.start(start_msg)

    def start(self, msg: str) -> None:
        _emit(self.stage, "▶", msg)

    def ok(self, msg: str) -> None:
        _emit(self.stage, "✔", f"{msg} · {self._elapsed()}")
        self._done = True

    def fail(self, msg: str) -> None:
        _emit(self.stage, "✖", f"{msg} · {self._elapsed()}")
        self._done = True

    def info(self, msg: str) -> None:
        _emit(self.stage, "→", msg)

    def _elapsed(self) -> str:
        return _c(DIM, f"{(time.perf_counter() - self._t0) * 1000:.0f} ms")

    def __enter__(self) -> "Stage":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._done:
            if exc_type is None:
                _emit(self.stage, "✔", self._elapsed())
            else:
                _emit(self.stage, "✖", f"{exc_type.__name__}: {exc} · {self._elapsed()}")