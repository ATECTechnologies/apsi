Apsi @V@ for Linux — developed by ATEC — https://apsi.dev
============================================================

Install (any distribution, no sudo):
    sh ApsiSetup-@V@-linux.sh
then open a new terminal and run:
    apsi version
    apsi doctor
    apsi new hello && cd hello && apsi run
    apsi install NAME            libraries from https://apsi.dev

Other ways to install:
    sudo apt install ./apsi_@V@_all.deb                 Debian / Ubuntu
    pip install apsi-@V@-py3-none-any.whl               any Python 3.12 or newer

Publish to the website from here (administrators; token with the "release" permission):
    apsi lib login --token apsi_...
    apsi release upload ApsiSetup-@V@-linux.sh apsi_@V@_all.deb apsi-@V@-py3-none-any.whl --version @V@
    apsi release list          apsi release delete ID

Files:
    ApsiSetup-@V@-linux.sh       self-extracting installer (uses the system Python 3.12+)
    apsi_@V@_all.deb             Debian / Ubuntu package
    apsi-@V@-py3-none-any.whl    pip package
    source/apsi-@V@-source.zip   complete source code, documentation, tests and examples
    LICENSE, CHANGELOG.md, INSTALL-LINUX.md
    gpu-tuner/                     Apsi GPU Tuner: Apply / Restore for Windows and Linux (best power limit for training)
    SHA256SUMS.txt                 checksums of every file
