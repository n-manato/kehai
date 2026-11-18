"""Put back the space between Latin sentences that arrive glued together.

Two sources glue them: whisper's segments come back joined without one
("Bye.All you.", "...UNO AI goals?Thank you.Okay, bye."), and replies were
stored piece by piece without one ("Mm,I'm fine.Same as always"). The stored
replies are replayed to the model as its own past turns, and on 2026-10-23 it
had started writing new ones the same way. Applied to what is transcribed, to
what is spoken, and to the history as it is loaded.

Conservative on purpose: after . ! ? only before a capital (so "weather.com",
"3.5" and "e.g." are left alone); after , ; : before any letter. Quotes are not
touched — a closing one is indistinguishable from an opening one here. CJK text
has no Latin punctuation and passes through unchanged.
"""

import re

_GLUED = re.compile(
    r"(?<=[a-z0-9'’)])[.!?](?=[A-Z])|(?<=[A-Za-z0-9'’)])[,;:](?=[A-Za-z])"
)


def unglue(text: str) -> str:
    if not text:
        return text
    return _GLUED.sub(lambda m: m.group(0) + " ", text)
