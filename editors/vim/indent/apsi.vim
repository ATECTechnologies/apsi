" Indentation for Apsi: indent after a line ending in ':', dedent after return/break/continue/pass
if exists('b:did_indent')
  finish
endif
let b:did_indent = 1
setlocal indentexpr=ApsiIndent(v:lnum)
setlocal indentkeys+=<:>,=elif,=else
setlocal autoindent nolisp nosmartindent
let b:undo_indent = 'setlocal indentexpr< indentkeys< autoindent< lisp< smartindent<'

if exists('*ApsiIndent')
  finish
endif

function! ApsiIndent(lnum) abort
  let prev = prevnonblank(a:lnum - 1)
  if prev == 0
    return 0
  endif
  let ptext = substitute(getline(prev), '\s*#.*$', '', '')
  let ind = indent(prev)
  let cur = getline(a:lnum)
  " open brackets continue the expression one level deeper
  let opens = len(substitute(ptext, '[^([{]', '', 'g')) - len(substitute(ptext, '[^)\]}]', '', 'g'))
  if opens < 0
    " the previous line closed a bracket opened earlier: go back to that statement's indent
    let bal = opens
    let l = prevnonblank(prev - 1)
    while l > 0 && bal < 0
      let t = substitute(getline(l), '\s*#.*$', '', '')
      let bal += len(substitute(t, '[^([{]', '', 'g')) - len(substitute(t, '[^)\]}]', '', 'g'))
      let start = l
      let l = prevnonblank(l - 1)
    endwhile
    if exists('start')
      let ind = indent(start)
      let ptext = substitute(getline(start), '\s*#.*$', '', '') . ptext
    endif
  endif
  if ptext =~# ':\s*$' || opens > 0
    let ind += shiftwidth()
  elseif ptext =~# '^\s*\%(return\|break\|continue\|pass\)\>'
    let ind -= shiftwidth()
  endif
  if cur =~# '^\s*\%(elif\|else\)\>'
    " line up with the matching if
    " line up with the nearest enclosing if/elif above the block that just ended
    let body = indent(prev)
    let l = prev
    while l > 0
      if getline(l) =~# '^\s*\%(if\|elif\)\>' && indent(l) < body
        return indent(l)
      endif
      let l = prevnonblank(l - 1)
    endwhile
    return max([ind - shiftwidth(), 0])
  endif
  return max([ind, 0])
endfunction
