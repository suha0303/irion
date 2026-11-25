[Setup]
AppId=ARION.LocalFlight
AppName=ARION
AppVersion={#AppVersion}
DefaultDirName={localappdata}\Programs\ARION
DefaultGroupName=ARION
OutputDir={#AppOutput}
OutputBaseFilename={#AppName}
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
Compression=lzma2/fast
SolidCompression=yes
UninstallDisplayIcon={app}\ARION.exe

[Files]
Source: "{#AppSource}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\ARION"; Filename: "{app}\ARION.exe"

[Run]
Filename: "{app}\ARION.exe"; Description: "Launch ARION"; Flags: nowait postinstall skipifsilent
