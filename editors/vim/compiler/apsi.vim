" :make checks the current Apsi file and fills the quickfix list
if exists('current_compiler')
  finish
endif
let current_compiler = 'apsi'
if exists(':CompilerSet') != 2
  command -nargs=* CompilerSet setlocal <args>
endif
CompilerSet makeprg=apsi\ check\ %:S
CompilerSet errorformat=%f:%l:%c:\ %m,%-G%.%#
