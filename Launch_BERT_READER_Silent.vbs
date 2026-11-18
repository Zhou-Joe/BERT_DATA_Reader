' ============================================================
'  BERT_READER X - silent launcher (no console flash at all)
' ============================================================
Set shell = CreateObject("WScript.Shell")
shell.CurrentDirectory = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)
shell.Run """C:\Users\ZC\ZCodeProject\accdata\.venv\Scripts\pythonw.exe"" main.py", 0, False
