; Apsi 1.10.0 — Windows 10/11 x64 installer (per-user, no admin rights needed)
; Built by packaging/build_all.sh with NSIS (makensis), which runs on Linux or Windows.
Unicode true
!include "MUI2.nsh"
!include "WinVer.nsh"
!include "x64.nsh"
!include "StrFunc.nsh"
${StrStr}
${UnStrRep}

!ifndef VERSION
  !define VERSION "1.10.0"
!endif
!ifndef PAYLOAD
  !define PAYLOAD "..\..\build\windows\Apsi"
!endif
!ifndef OUTFILE
  !define OUTFILE "..\..\dist\ApsiSetup-${VERSION}-x64.exe"
!endif

Name "Apsi ${VERSION}"
OutFile "${OUTFILE}"
InstallDir "$LOCALAPPDATA\Programs\Apsi"
InstallDirRegKey HKCU "Software\Apsi" "InstallDir"
RequestExecutionLevel user
SetCompressor /SOLID lzma
BrandingText "Apsi ${VERSION}"
VIProductVersion "${VERSION}.0"
VIAddVersionKey "ProductName" "Apsi"
VIAddVersionKey "FileDescription" "Apsi programming language installer"
VIAddVersionKey "ProductVersion" "${VERSION}"
VIAddVersionKey "FileVersion" "${VERSION}"
VIAddVersionKey "LegalCopyright" "Apsi authors"

!define MUI_ABORTWARNING
!define MUI_ICON "..\..\assets\brand\apsi.ico"
!define MUI_UNICON "..\..\assets\brand\apsi.ico"
!define MUI_WELCOMEFINISHPAGE_BITMAP "..\..\assets\brand\nsis-welcome.bmp"
!define MUI_UNWELCOMEFINISHPAGE_BITMAP "..\..\assets\brand\nsis-welcome.bmp"
!define MUI_HEADERIMAGE
!define MUI_HEADERIMAGE_RIGHT
!define MUI_HEADERIMAGE_BITMAP "..\..\assets\brand\nsis-header.bmp"
!define MUI_WELCOMEPAGE_TITLE "Welcome to Apsi ${VERSION}"
!define MUI_WELCOMEPAGE_TEXT "Setup will install Apsi ${VERSION}, a programming language for AI, physics and servers.$\r$\n$\r$\nApsi installs for your user account only and does not need administrator rights.$\r$\n$\r$\nSelect Next to continue."
!define MUI_FINISHPAGE_RUN "$INSTDIR\apsi-repl.cmd"
!define MUI_FINISHPAGE_RUN_TEXT "Open the Apsi REPL"
!define MUI_FINISHPAGE_SHOWREADME "$INSTDIR\README.md"
!define MUI_FINISHPAGE_SHOWREADME_TEXT "Show README"
!define MUI_FINISHPAGE_SHOWREADME_NOTCHECKED
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "English"

Function .onInit
  ${IfNot} ${RunningX64}
    MessageBox MB_ICONSTOP "Apsi requires 64-bit Windows 10 or newer."
    Abort
  ${EndIf}
  ${IfNot} ${AtLeastWin10}
    MessageBox MB_ICONSTOP "Apsi requires Windows 10 or newer."
    Abort
  ${EndIf}
FunctionEnd

Section "Apsi" SecMain
  SectionIn RO
  SetOutPath "$INSTDIR"
  RMDir /r "$INSTDIR\runtime"
  File /r "${PAYLOAD}\*.*"
  WriteRegStr HKCU "Software\Apsi" "InstallDir" "$INSTDIR"
  WriteRegStr HKCU "Software\Apsi" "Version" "${VERSION}"

  ; --- add to the user PATH (HKCU\Environment) if missing
  ReadRegStr $0 HKCU "Environment" "Path"
  ${StrStr} $1 "$0" "$INSTDIR"
  ${If} $1 == ""
    ${If} $0 == ""
      WriteRegExpandStr HKCU "Environment" "Path" "$INSTDIR"
    ${Else}
      WriteRegExpandStr HKCU "Environment" "Path" "$0;$INSTDIR"
    ${EndIf}
    SendMessage ${HWND_BROADCAST} ${WM_SETTINGCHANGE} 0 "STR:Environment" /TIMEOUT=5000
  ${EndIf}

  ; --- .apsi file type: double-click = run, right-click Edit = Notepad
  WriteRegStr HKCU "Software\Classes\.apsi" "" "Apsi.Source"
  WriteRegStr HKCU "Software\Classes\.apsi" "Content Type" "text/plain"
  WriteRegStr HKCU "Software\Classes\Apsi.Source" "" "Apsi source file"
  WriteRegStr HKCU "Software\Classes\Apsi.Source\DefaultIcon" "" "$INSTDIR\brand\apsi-file.ico"
  WriteRegStr HKCU "Software\Classes\Apsi.Source\shell\open" "" "Run with Apsi"
  WriteRegStr HKCU "Software\Classes\Apsi.Source\shell\open\command" "" '"$INSTDIR\apsi-run.cmd" "%1"'
  WriteRegStr HKCU "Software\Classes\Apsi.Source\shell\edit" "" "Edit"
  WriteRegStr HKCU "Software\Classes\Apsi.Source\shell\edit\command" "" 'notepad.exe "%1"'
  WriteRegStr HKCU "Software\Classes\Apsi.Source\shell\test" "" "Test with Apsi"
  WriteRegStr HKCU "Software\Classes\Apsi.Source\shell\test\command" "" '"$INSTDIR\apsi-run.cmd" --test "%1"'
  System::Call 'shell32::SHChangeNotify(i 0x08000000, i 0, p 0, p 0)'

  ; --- Start menu
  CreateDirectory "$SMPROGRAMS\Apsi"
  CreateShortcut "$SMPROGRAMS\Apsi\Apsi REPL.lnk" "$INSTDIR\apsi-repl.cmd"
  CreateShortcut "$SMPROGRAMS\Apsi\Apsi Examples.lnk" "$INSTDIR\examples"
  CreateShortcut "$SMPROGRAMS\Apsi\Apsi Documentation.lnk" "$INSTDIR\docs"
  CreateShortcut "$SMPROGRAMS\Apsi\Uninstall Apsi.lnk" "$INSTDIR\Uninstall.exe"

  ; --- Apps & Features entry
  WriteUninstaller "$INSTDIR\Uninstall.exe"
  !define UNKEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\Apsi"
  WriteRegStr HKCU "${UNKEY}" "DisplayName" "Apsi ${VERSION}"
  WriteRegStr HKCU "${UNKEY}" "DisplayVersion" "${VERSION}"
  WriteRegStr HKCU "${UNKEY}" "Publisher" "Apsi"
  WriteRegStr HKCU "${UNKEY}" "InstallLocation" "$INSTDIR"
  WriteRegStr HKCU "${UNKEY}" "DisplayIcon" "$INSTDIR\brand\apsi.ico"
  WriteRegStr HKCU "${UNKEY}" "UninstallString" '"$INSTDIR\Uninstall.exe"'
  WriteRegDWORD HKCU "${UNKEY}" "NoModify" 1
  WriteRegDWORD HKCU "${UNKEY}" "NoRepair" 1

  ; --- quick self-test so a broken install is reported immediately
  nsExec::ExecToLog '"$INSTDIR\apsi.cmd" version'
  Pop $2
  ${If} $2 != 0
    MessageBox MB_ICONEXCLAMATION "Apsi was copied but the self-test failed (code $2). Run 'apsi doctor' in a new terminal."
  ${EndIf}
SectionEnd

Section "Uninstall"
  ReadRegStr $0 HKCU "Environment" "Path"
  ${UnStrRep} $0 "$0" ";$INSTDIR" ""
  ${UnStrRep} $0 "$0" "$INSTDIR;" ""
  ${UnStrRep} $0 "$0" "$INSTDIR" ""
  WriteRegExpandStr HKCU "Environment" "Path" "$0"
  SendMessage ${HWND_BROADCAST} ${WM_SETTINGCHANGE} 0 "STR:Environment" /TIMEOUT=5000
  DeleteRegKey HKCU "Software\Classes\Apsi.Source"
  DeleteRegKey HKCU "Software\Classes\.apsi"
  DeleteRegKey HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\Apsi"
  DeleteRegKey HKCU "Software\Apsi"
  RMDir /r "$SMPROGRAMS\Apsi"
  RMDir /r "$INSTDIR\runtime"
  RMDir /r "$INSTDIR\examples"
  RMDir /r "$INSTDIR\docs"
  RMDir /r "$INSTDIR\editors"
  Delete "$INSTDIR\*.cmd"
  Delete "$INSTDIR\*.md"
  Delete "$INSTDIR\*.vsix"
  Delete "$INSTDIR\Uninstall.exe"
  RMDir "$INSTDIR"
  System::Call 'shell32::SHChangeNotify(i 0x08000000, i 0, p 0, p 0)'
SectionEnd
