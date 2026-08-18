# Trae los datos frescos de GitHub (refresh diario de las 06:00 UTC).
# Pensado para correr desde el Programador de tareas de Windows.
#
# Seguridad: si hay cambios locales sin commitear, NO hace pull (evita
# conflictos o pisar trabajo en curso) y lo deja anotado en auto_pull.log.

$repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $repo
$log = Join-Path $repo "auto_pull.log"
$stamp = Get-Date -Format "yyyy-MM-dd HH:mm"

$dirty = git status --porcelain
if ($dirty) {
    Add-Content $log "$stamp | SKIP: hay cambios locales sin commitear, no se hace pull"
    exit 0
}

$result = git pull --ff-only 2>&1 | Out-String
Add-Content $log "$stamp | $($result.Trim())"
