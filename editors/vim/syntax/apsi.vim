" Vim syntax file
" Language: Apsi (https://github.com/apsi-lang/apsi)
if exists('b:current_syntax')
  finish
endif
let s:cpo_save = &cpo
set cpo&vim

syn keyword apsiStatement   let var nextgroup=apsiVarName skipwhite
syn keyword apsiFunction    fn nextgroup=apsiFuncName skipwhite
syn keyword apsiStructKw    struct nextgroup=apsiStructName skipwhite
syn keyword apsiConditional if elif else
syn keyword apsiRepeat      while for in break continue parallel
syn keyword apsiStatement   return pass assert
syn keyword apsiInclude     use import as
syn keyword apsiOperatorKw  and or not
syn keyword apsiBoolean     true false
syn keyword apsiNone        none
syn keyword apsiEvolve      evolve over step using tol every into
syn keyword apsiConstant    pi tau inf nan
syn keyword apsiSelf        self
syn match   apsiTest        "^\s*\zstest\ze\s\+\""
syn match   apsiEvolve      "\<record\ze\s\+\%(every\|into\)\>"

syn match   apsiVarName     "\h\w*" contained
syn match   apsiFuncName    "\h\w*" contained
syn match   apsiStructName  "\h\w*" contained
syn match   apsiDecorator   "^\s*\zs@\h\w*\%(\.\h\w*\)*"
syn match   apsiCall        "\<\h\w*\ze\s*("
syn match   apsiDeriv       "^\s*\h\w*\zs'\ze\s*="

syn keyword apsiBuiltin any all divmod pow ord chr dict set tuple repr hex bin oct next iter format callable getattr hasattr fsum
syn keyword apsiBuiltin abs acos approx arange argmax argmin args asin atan atan2 bool cbrt ceil chars clip
syn keyword apsiBuiltin cload clock concat cos cosh cross cumsum dequantize derivative dim dot einsum enumerate
syn keyword apsiBuiltin error exact exists exit exp eye filter flatten float floor format_info from_py from_torch
syn keyword apsiBuiltin full gelu getenv grad hessian http_get http_post_json input int jacobian join json_parse
syn keyword apsiBuiltin json_str keys laplace len library_path linspace list load_csv load_json log log10 log2
syn keyword apsiBuiltin log_softmax magnitude map matmul max mean measure memo meshgrid min nbytes ndim nominal
syn keyword apsiBuiltin norm ones ones_like outer param_count permissions permutation platform pmap precision
syn keyword apsiBuiltin print prod pyimport qmatmul quant_error quantize rand randint randn range read_text
syn keyword apsiBuiltin record relative_uncertainty relu reshape reversed roll round round_to save_csv save_json
syn keyword apsiBuiltin security_policy seed shape sigma sigmoid sin sinh softmax softplus sort sorted sqrt stack
syn keyword apsiBuiltin std stop_gradient str sum tan tanh tensor testing timed to_float to_py to_torch transpose
syn keyword apsiBuiltin tree_map type unique unit value_and_grad where write_text zeros zeros_like zip

syn keyword apsiType Float Int Bool Str List Tuple Dict Any Fn Record Tensor Vec Vec2 Vec3 Vec4 Mat Matrix
syn keyword apsiType Dimensionless Angle Length Mass Time Current Temperature Amount LuminousIntensity Area Volume
syn keyword apsiType Velocity Speed Acceleration Jerk Force Energy Work Torque Power Pressure Stress Momentum
syn keyword apsiType Impulse AngularMomentum Action Frequency AngularVelocity Density Charge Voltage Resistance
syn keyword apsiType Capacitance Inductance MagneticField MagneticFlux ElectricField Entropy HeatCapacity Viscosity
syn keyword apsiType Diffusivity KinematicViscosity WaveNumber Concentration MolarMass ThermalConductivity SpringConstant
syn keyword apsiDtype f16 f32 f64 bf16 i32 i64 int8 int4 uint8 fp8 bf8 fp6 fp4 fp8_e4m3 fp8_e5m2 fp6_e3m2 fp6_e2m3 fp4_e2m1

" numbers with units: 5kg, 9.81[m/s^2], and unit conversions: as [km]
syn match   apsiNumber   "\<\d[0-9_]*\%(\.\d[0-9_]*\)\=\%([eE][+-]\=\d\+\)\=" nextgroup=apsiUnitSuffix,apsiUnitBracket
syn match   apsiUnitSuffix  "[A-Za-zµμΩÅ°][A-Za-z0-9µμΩÅ]*" contained
syn region  apsiUnitBracket matchgroup=apsiUnitDelim start="\[" end="\]" contained oneline contains=apsiUnitOp
syn match   apsiUnitOp   "[*/^·]\|\*\*\|-\=\d\+" contained
syn region  apsiUnitCast matchgroup=apsiOperatorKw start="\<as\s*\[" matchgroup=apsiUnitDelim end="\]" oneline contains=apsiUnitOp

syn match   apsiOperator "|>\|->\|=>\|±\|+/-\|\.\.=\=\|\~=\|==\|!=\|<=\|>=\|\*\*\|//\|[-+*/%@^<>=]"

syn match   apsiEscape   "\\[ntr\\\"0]" contained
syn match   apsiEscape   "{{\|}}" contained
syn region  apsiString   start=+"+ skip=+\\"+ end=+"+ oneline contains=apsiEscape
syn region  apsiString   start=+"""+ end=+"""+ contains=apsiEscape
syn region  apsiFString  start=+f"+ skip=+\\"+ end=+"+ oneline contains=apsiEscape,apsiInterp
syn region  apsiFString  start=+f"""+ end=+"""+ contains=apsiEscape,apsiInterp
syn region  apsiInterp   matchgroup=apsiInterpDelim start="{\ze[^{]" end="}" contained contains=ALLBUT,apsiVarName,apsiFuncName,apsiStructName,apsiUnitSuffix,apsiUnitBracket,apsiUnitOp,apsiEscape

syn keyword apsiTodo     TODO FIXME NOTE HACK XXX contained
syn match   apsiComment  "#.*$" contains=apsiTodo,@Spell

hi def link apsiStatement   Statement
hi def link apsiFunction    Keyword
hi def link apsiStructKw    Structure
hi def link apsiConditional Conditional
hi def link apsiRepeat      Repeat
hi def link apsiInclude     Include
hi def link apsiOperatorKw  Operator
hi def link apsiBoolean     Boolean
hi def link apsiNone        Constant
hi def link apsiEvolve      Special
hi def link apsiTest        Special
hi def link apsiConstant    Constant
hi def link apsiSelf        Identifier
hi def link apsiFuncName    Function
hi def link apsiStructName  Type
hi def link apsiVarName     Identifier
hi def link apsiDecorator   PreProc
hi def link apsiCall        Function
hi def link apsiDeriv       Special
hi def link apsiBuiltin     Function
hi def link apsiType        Type
hi def link apsiDtype       Type
hi def link apsiNumber      Number
hi def link apsiUnitSuffix  Special
hi def link apsiUnitBracket Special
hi def link apsiUnitCast    Special
hi def link apsiUnitDelim   Delimiter
hi def link apsiUnitOp      Special
hi def link apsiOperator    Operator
hi def link apsiString      String
hi def link apsiFString     String
hi def link apsiEscape      SpecialChar
hi def link apsiInterpDelim SpecialChar
hi def link apsiComment     Comment
hi def link apsiTodo        Todo

let b:current_syntax = 'apsi'
let &cpo = s:cpo_save
unlet s:cpo_save
