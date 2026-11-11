"""Builds the four Apsi GPU Tuner files (Windows apply / restore, Linux apply / restore) from the two templates."""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "dist")
os.makedirs(OUT, exist_ok=True)
ps = open(os.path.join(HERE, "template.ps1"), encoding="utf-8").read()
sh = open(os.path.join(HERE, "template.sh"), encoding="utf-8").read()
WHAT = {"apply": "sets every NVIDIA card to the recommended power limit for long AI training (cooler, steady, "
                 "about the same speed) and keeps it after restarts",
        "restore": "removes everything the Apsi GPU Tuner set and puts every card back to its factory default"}
for mode, name in (("apply", "Apsi-GPU-Tuner-Apply"), ("restore", "Apsi-GPU-Tuner-Restore")):
    bat = ("@echo off\r\nrem Apsi GPU Tuner: " + WHAT[mode] + ".\r\n"
           "rem Double-click it; Windows asks once for administrator rights (setting a power limit needs them).\r\n"
           "net session >nul 2>&1\r\nif %errorlevel% neq 0 (\r\n"
           "  powershell -NoProfile -Command \"Start-Process -FilePath $env:TUNER_SELF -Verb RunAs\"\r\n  exit /b\r\n)\r\n"
           .replace("$env:TUNER_SELF", "'%~f0'") +
           "set \"TUNER=%~f0\"\r\n"
           "powershell -NoProfile -ExecutionPolicy Bypass -Command \"$Mode='" + mode + "'; "
           "iex ((Get-Content -LiteralPath $env:TUNER -Raw) -split '##[ ]PS[ ]##')[1]\"\r\n"
           "echo.\r\npause\r\nexit /b\r\n## PS ##\r\n" + ps.replace("\r\n", "\n").replace("\n", "\r\n"))
    open(os.path.join(OUT, name + ".bat"), "w", encoding="ascii", newline="").write(bat)
    lin = name.lower().replace("apsi-gpu-tuner-", "apsi-gpu-tuner-") + "-linux.sh"
    open(os.path.join(OUT, lin), "w", encoding="ascii", newline="\n").write(
        "#!/bin/sh\n# Apsi GPU Tuner: " + WHAT[mode] + ".\n# Run: sh " + lin + "   (asks for sudo)\n"
        "MODE=${MODE:-" + mode + "}\n" + sh)
print("\n".join(sorted(os.listdir(OUT))))
