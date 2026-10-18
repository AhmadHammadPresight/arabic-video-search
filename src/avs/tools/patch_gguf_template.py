"""Make a GGUF whose chat template uses {% raw %} load in Ollama, for the query model comparison only.

    python -m avs.tools.patch_gguf_template SOURCE.gguf OUT.gguf

Fanar 2's published GGUF wraps its tool-calling instructions in {% raw %} ... {% endraw %}, which
the template engine in Ollama's llama.cpp runner rejects ("Unknown statement: raw"), so the model
does not load. The text inside has no Jinja delimiters, so the two tags change nothing; this
writes a copy of the GGUF with each tag replaced by a Jinja comment of the same length, which
renders exactly the same text. The weights and every other byte stay the same.
The copy is a copy-on-write clone where the disk supports it; the source is never modified.

Find the source with `ollama show --modelfile <model>` (its FROM line), then import the copy with
`ollama create <name> -f Modelfile` using the published template and stop parameter.
"""

from __future__ import annotations

import argparse
import mmap
import shutil
import subprocess
import sys
from pathlib import Path

REPLACEMENTS = ((b"{% raw %}", b"{#     #}"), (b"{% endraw %}", b"{#        #}"))
# The metadata, with the vocabulary and the chat template, comes before the weights in a GGUF.
METADATA_WINDOW = 64 * 1024 * 1024


def patch(source: Path, out: Path) -> list[tuple[str, int]]:
    with source.open("rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"{source} is not a GGUF file")
    if subprocess.run(["cp", "-c", str(source), str(out)], capture_output=True).returncode != 0:
        shutil.copyfile(source, out)
    done = []
    with out.open("r+b") as f, mmap.mmap(f.fileno(), 0) as m:
        window = min(len(m), METADATA_WINDOW)
        for old, new in REPLACEMENTS:
            assert len(old) == len(new)
            at = m.find(old, 0, window)
            if at < 0 or m.find(old, at + 1, window) >= 0:
                out.unlink()
                raise ValueError(f"expected exactly one {old.decode()} in the metadata of {source}")
            m[at:at + len(old)] = new
            done.append((old.decode(), at))
        m.flush()
    return done


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args(argv)
    for tag, at in patch(args.source, args.out):
        print(f"replaced {tag} at byte {at}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
