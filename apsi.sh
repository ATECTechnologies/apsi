#!/usr/bin/env sh
# run Apsi without installing:  ./apsi.sh run examples/01_hello.apsi
DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHONPATH="$DIR${PYTHONPATH:+:$PYTHONPATH}" exec python3 -m apsi "$@"
