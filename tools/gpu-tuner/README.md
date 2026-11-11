# Apsi GPU Tuner

`python build.py [OUT]` writes the four files published on the site's download page:

- `Apsi-GPU-Tuner-Apply.bat` / `Apsi-GPU-Tuner-Restore.bat` (Windows: a batch header that asks for administrator
  rights, then runs the PowerShell in `template.ps1`)
- `apsi-gpu-tuner-apply-linux.sh` / `apsi-gpu-tuner-restore-linux.sh` (Linux: `template.sh`, asks for sudo)

Apply sets each NVIDIA card to the recommended sustained-training power limit (see docs/GPU.md, section 4) with
`nvidia-smi -pl`, keeps it after restarts (Task Scheduler task "Apsi GPU Tuner" / systemd unit apsi-gpu-tuner), and
runs `apsi gpu setup` when Apsi is installed. Restore removes the task / unit and sets every card back to its default.
