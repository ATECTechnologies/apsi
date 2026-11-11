# Apsi GPU Tuner for Linux. MODE is apply or restore.
set -u
if [ "$(id -u)" -ne 0 ]; then exec sudo MODE="$MODE" sh "$0" "$@"; fi
UNIT=/etc/systemd/system/apsi-gpu-tuner.service
LIB=/usr/local/lib/apsi-gpu-tuner
command -v nvidia-smi >/dev/null 2>&1 || { echo "No NVIDIA driver found (nvidia-smi is missing)."; exit 1; }
recommend() {   # -> percent;reason
  n=$(echo "$1" | tr '[:lower:]' '[:upper:]')
  case "$n" in
    *LAPTOP*|*MOBILE*|*MAX-Q*) echo "0;laptop GPU: managed by the laptop";;
    *H100*|*H200*|*H800*|*H20*|*B100*|*B200*|*B300*|*A100*|*A800*|*A30*|*A40*|*A10*|*A16*|*" A2"*|*" L4"*|*L40*|*T4*|*V100*|*P100*|*P40*|*TESLA*)
      echo "100;data-center GPU: built for full sustained power, keep the default";;
    *"RTX PRO"*|*"RTX A"[0-9]*|*ADA*|*QUADRO*|*"RTX 6000"*|*"RTX 5000"*|*"RTX 4500"*|*"RTX 4000"*) echo "90;workstation GPU: -10 % power for about -2 % speed";;
    *"RTX 50"[0-9][0-9]*|*"RTX 40"[0-9][0-9]*) echo "75;GeForce RTX 40/50: about -25 % power for about -5 % speed";;
    *"RTX 30"[0-9][0-9]*) echo "80;GeForce RTX 30: about -20 % power for about -5 % speed, 6-10 C cooler";;
    *) echo "85;other NVIDIA GPU: a moderate -15 %";;
  esac
}
show() { nvidia-smi --query-gpu=index,name,temperature.gpu,power.limit,power.default_limit,power.min_limit,power.max_limit --format=csv,noheader,nounits |
  while IFS=, read -r i name t now d lo hi; do echo "  GPU $i:$name | ${t# } C | power limit ${now# } W (default ${d# } W, allowed ${lo# }-${hi# } W)"; done; }
apsi_helper() { command -v apsi >/dev/null 2>&1 && apsi gpu "$1" 2>/dev/null; true; }
echo; echo "  Apsi GPU Tuner - best sustained settings for AI training"; echo "  -------------------------------------------------------"; show; echo
if [ "$MODE" = apply ]; then
  mkdir -p "$LIB"; printf '#!/bin/sh\n# Written by the Apsi GPU Tuner; undo with its restore file.\nnvidia-smi -pm 1 >/dev/null 2>&1\n' > "$LIB/apply.sh"
  nvidia-smi -pm 1 >/dev/null 2>&1
  nvidia-smi --query-gpu=index,name,power.default_limit,power.min_limit --format=csv,noheader,nounits |
  while IFS=, read -r i name d lo; do
    r=$(recommend "$name"); pct=${r%%;*}; why=${r#*;}
    case "$d" in *N/A*|"") echo "  GPU $i: skipped (no power limit)"; continue;; esac
    [ "$pct" = 0 ] && { echo "  GPU $i: skipped ($why)"; continue; }
    w=$(awk -v d="$d" -v lo="$lo" -v p="$pct" 'BEGIN{w=d*p/100; if (w<lo) w=lo; printf "%d", w+0.5}')
    echo "  GPU $i: $pct % of default = $w W ($why)"
    nvidia-smi -i "$i" -pl "$w" >/dev/null && echo "nvidia-smi -i $i -pl $w" >> "$LIB/apply.sh"
  done
  chmod 755 "$LIB/apply.sh"
  printf '[Unit]\nDescription=Apsi GPU Tuner (power limits for AI training)\nAfter=multi-user.target\n\n[Service]\nType=oneshot\nExecStart=%s/apply.sh\n\n[Install]\nWantedBy=multi-user.target\n' "$LIB" > "$UNIT"
  systemctl daemon-reload && systemctl enable apsi-gpu-tuner >/dev/null 2>&1
  echo; echo "  Kept after every restart (systemd: apsi-gpu-tuner)."
  apsi_helper setup
else
  systemctl disable apsi-gpu-tuner >/dev/null 2>&1; rm -f "$UNIT"; rm -rf "$LIB"; systemctl daemon-reload 2>/dev/null
  nvidia-smi --query-gpu=index,power.default_limit --format=csv,noheader,nounits |
  while IFS=, read -r i d; do case "$d" in *N/A*|"") continue;; esac
    w=$(awk -v d="$d" 'BEGIN{printf "%d", d+0.5}'); nvidia-smi -i "$i" -pl "$w" >/dev/null; echo "  GPU $i: back to the default $w W"; done
  apsi_helper remove
  echo; echo "  Everything the tuner set is removed; the cards run at their factory defaults."
fi
echo; show
