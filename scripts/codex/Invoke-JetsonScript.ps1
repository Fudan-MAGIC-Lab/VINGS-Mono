param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$ScriptPath,

    [string]$HostName = "jetson-codex",
    [string]$RepoPath = "/home/jetson/VINGS-Mono",
    [string]$CondaRoot = "/home/jetson/miniconda3",
    [string]$CondaEnv = "vings_jetson",
    [switch]$UseConda,
    [switch]$KeepRemoteScript
)

if (-not (Test-Path -LiteralPath $ScriptPath)) {
    Write-Error "Script not found: $ScriptPath"
    exit 2
}

$scriptName = [System.IO.Path]::GetFileName($ScriptPath)
$stamp = Get-Date -Format "yyyyMMddHHmmssfff"
$localTemp = Join-Path $env:TEMP "codex-jetson-$stamp-$scriptName"
$remoteTemp = "/tmp/codex-jetson-$stamp-$scriptName"

try {
    $content = Get-Content -LiteralPath $ScriptPath -Raw
    $content = $content -replace "`r`n", "`n"
    $content = $content -replace "`r", "`n"
    $utf8NoBom = New-Object System.Text.UTF8Encoding -ArgumentList $false
    [System.IO.File]::WriteAllText($localTemp, $content, $utf8NoBom)

    scp $localTemp "${HostName}:$remoteTemp"
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

    $remoteCommand = "cd '$RepoPath' && "
    if ($UseConda) {
        $remoteCommand += "source '$CondaRoot/etc/profile.d/conda.sh' && conda activate '$CondaEnv' && "
    }
    $remoteCommand += "bash '$remoteTemp'"

    ssh $HostName $remoteCommand
    $exitCode = $LASTEXITCODE

    if (-not $KeepRemoteScript) {
        ssh $HostName "rm -f '$remoteTemp'" | Out-Null
    }

    exit $exitCode
}
finally {
    Remove-Item -LiteralPath $localTemp -Force -ErrorAction SilentlyContinue
}
