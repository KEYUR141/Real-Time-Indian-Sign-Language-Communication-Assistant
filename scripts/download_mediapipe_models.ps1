[CmdletBinding()]
param(
    [string]$OutputDirectory = (Join-Path $PSScriptRoot '..\models\mediapipe'),
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$OutputDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null

$models = @(
    [pscustomobject]@{
        Name = 'hand_landmarker.task'
        Url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
        Sha256 = 'fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1'
        SizeBytes = 7819105
    },
    [pscustomobject]@{
        Name = 'pose_landmarker_lite.task'
        Url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task'
        Sha256 = '59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a'
        SizeBytes = 5777746
    }
)

function Get-Sha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

foreach ($model in $models) {
    $destination = Join-Path $OutputDirectory $model.Name
    $temporary = "$destination.download"

    if ((Test-Path -LiteralPath $destination) -and -not $Force) {
        $existingHash = Get-Sha256 $destination
        if ($existingHash -eq $model.Sha256) {
            Write-Host "Verified existing model: $($model.Name)"
            continue
        }

        throw "Existing model has an unexpected SHA-256 hash: $destination. Use -Force to replace it."
    }

    Remove-Item -LiteralPath $temporary -Force -ErrorAction SilentlyContinue
    Write-Host "Downloading $($model.Name)..."
    & curl.exe -L --fail --retry 10 --retry-all-errors --retry-delay 5 `
        --output $temporary $model.Url

    if ($LASTEXITCODE -ne 0) {
        throw "Download failed with curl exit code $LASTEXITCODE`: $($model.Url)"
    }

    $downloadedFile = Get-Item -LiteralPath $temporary
    if ($downloadedFile.Length -ne $model.SizeBytes) {
        throw "Size mismatch for $($model.Name). Expected $($model.SizeBytes), got $($downloadedFile.Length)."
    }

    $downloadedHash = Get-Sha256 $temporary
    if ($downloadedHash -ne $model.Sha256) {
        throw "SHA-256 mismatch for $($model.Name). Expected $($model.Sha256), got $downloadedHash."
    }

    Move-Item -LiteralPath $temporary -Destination $destination -Force
    Write-Host "Installed and verified: $destination"
}

Write-Host 'MediaPipe model setup complete.'

