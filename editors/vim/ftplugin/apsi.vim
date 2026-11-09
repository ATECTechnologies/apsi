" Apsi file settings
if exists('b:did_ftplugin')
  finish
endif
let b:did_ftplugin = 1

setlocal expandtab shiftwidth=4 softtabstop=4 tabstop=4
setlocal commentstring=#\ %s comments=:#
setlocal suffixesadd=.apsi
setlocal iskeyword+=_
compiler apsi

" <LocalLeader>r run · t test · c check · f format · i REPL
nnoremap <buffer> <silent> <LocalLeader>r :update<CR>:!apsi run %:p:S<CR>
nnoremap <buffer> <silent> <LocalLeader>t :update<CR>:!apsi test %:p:S<CR>
nnoremap <buffer> <silent> <LocalLeader>c :update<CR>:make<CR>
nnoremap <buffer> <silent> <LocalLeader>f :update<CR>:silent !apsi fmt %:p:S<CR>:edit!<CR>
if has('terminal') || has('nvim')
  nnoremap <buffer> <silent> <LocalLeader>i :terminal apsi repl<CR>
endif

let b:undo_ftplugin = 'setlocal et< sw< sts< ts< cms< com< sua< isk<'
      \ . '| silent! nunmap <buffer> <LocalLeader>r | silent! nunmap <buffer> <LocalLeader>t'
      \ . '| silent! nunmap <buffer> <LocalLeader>c | silent! nunmap <buffer> <LocalLeader>f'
      \ . '| silent! nunmap <buffer> <LocalLeader>i'
