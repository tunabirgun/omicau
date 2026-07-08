; Inno Setup script for omicau (Windows installer).
; Compile:  iscc packaging/omicau.iss   (produces packaging/Output/omicau-setup.exe)
; Sign the resulting installer and the inner omicau.exe with the same identity so
; SmartScreen reputation accrues to one certificate.

#define AppName "omicau"
#define AppVersion "0.1.0"
#define AppPublisher "Tuna Birgun"
#define AppURL "https://github.com/tunabirgun/omicau"

[Setup]
AppId={{9E4B2C10-0A7E-4C2E-9E9B-0MICAU000001}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
OutputDir=Output
OutputBaseFilename=omicau-setup-{#AppVersion}
Compression=lzma2/max
SolidCompression=yes
PrivilegesRequired=lowest
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"

[Files]
; The PyInstaller onedir output (dist/omicau) — copied whole.
Source: "..\dist\omicau\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\omicau.exe"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\omicau.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\omicau.exe"; Description: "Launch omicau"; Flags: nowait postinstall skipifsilent
