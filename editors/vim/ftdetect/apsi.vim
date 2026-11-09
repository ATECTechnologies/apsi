" Apsi files: *.apsi, and scripts starting with #!/usr/bin/env apsi
autocmd BufRead,BufNewFile *.apsi setfiletype apsi
autocmd BufRead,BufNewFile * if getline(1) =~# '^#!.*\<apsi\>' | setfiletype apsi | endif
autocmd BufRead,BufNewFile apsi.toml setfiletype toml
