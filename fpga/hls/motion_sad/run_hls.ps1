[CmdletBinding()]
param(
    [switch]$Synthesize
)

$ErrorActionPreference = 'Stop'

$vitisHls = 'D:\xilinx\Vitis_HLS\2023.1\bin\vitis_hls.bat'
if (-not (Test-Path -LiteralPath $vitisHls)) {
    throw "Vitis HLS 2023.1 was not found at $vitisHls"
}

$stageRoot = Join-Path $env:LOCALAPPDATA 'VINGS-Mono\hls'
$stage = Join-Path $stageRoot ('motion_sad_2023_1_' + [guid]::NewGuid().ToString('N'))
$originalStage = [Environment]::GetEnvironmentVariable(
    'MOTION_SAD_STAGE_DIR', 'Process')
$originalSynthesis = [Environment]::GetEnvironmentVariable(
    'MOTION_SAD_SYNTH', 'Process')
$pushedLocation = $false

New-Item -ItemType Directory -Path $stage -Force | Out-Null
$env:MOTION_SAD_STAGE_DIR = $stage
if ($Synthesize) {
    $env:MOTION_SAD_SYNTH = '1'
} else {
    Remove-Item Env:MOTION_SAD_SYNTH -ErrorAction SilentlyContinue
}

try {
    Push-Location $PSScriptRoot
    $pushedLocation = $true
    $output = & $vitisHls -f run_hls.tcl 2>&1
    $exitCode = $LASTEXITCODE
    $output | ForEach-Object { Write-Output $_ }
    $outputText = $output -join "`n"

    if ($exitCode -ne 0 -or $outputText -match '(?m)^ERROR:') {
        throw "Vitis HLS failed with exit code $exitCode"
    }
    if ($outputText -notmatch 'All motion_sad tests passed') {
        throw 'Vitis HLS did not report a passing C simulation.'
    }
    if ($Synthesize -and
        $outputText -notmatch 'Finished Command csynth_design') {
        throw 'Vitis HLS did not report completed C synthesis.'
    }
} finally {
    if ($pushedLocation) {
        Pop-Location
    }
    if ($null -eq $originalStage) {
        Remove-Item Env:MOTION_SAD_STAGE_DIR -ErrorAction SilentlyContinue
    } else {
        $env:MOTION_SAD_STAGE_DIR = $originalStage
    }
    if ($null -eq $originalSynthesis) {
        Remove-Item Env:MOTION_SAD_SYNTH -ErrorAction SilentlyContinue
    } else {
        $env:MOTION_SAD_SYNTH = $originalSynthesis
    }

    $resolvedRoot = [IO.Path]::GetFullPath($stageRoot)
    $resolvedStage = [IO.Path]::GetFullPath($stage)
    if (-not $resolvedStage.StartsWith(
            $resolvedRoot + [IO.Path]::DirectorySeparatorChar,
            [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove HLS stage outside $resolvedRoot"
    }
    if (Test-Path -LiteralPath $resolvedStage) {
        Remove-Item -LiteralPath $resolvedStage -Recurse -Force
    }
}
