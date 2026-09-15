[CmdletBinding()]
param(
    [string]$DataRoot = (Join-Path $PSScriptRoot '..\data'),
    [switch]$KeepArchives
)

$ErrorActionPreference = 'Stop'

$recordId = '4010759'
$rawDir = Join-Path $DataRoot 'raw\include'
$manifestDir = Join-Path $DataRoot 'manifests'
$downloadDir = Join-Path $DataRoot '.downloads\include'
$stateDir = Join-Path $DataRoot '.state\include'

@($rawDir, $manifestDir, $downloadDir, $stateDir) | ForEach-Object {
    New-Item -ItemType Directory -Path $_ -Force | Out-Null
}

function Get-Md5([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm MD5).Hash.ToLowerInvariant()
}

function Invoke-ResumableDownload([string]$Url, [string]$Destination) {
    & curl.exe -L --fail --retry 5 --retry-delay 5 -C - --output $Destination $Url
    if ($LASTEXITCODE -ne 0) {
        throw "Download failed with curl exit code $LASTEXITCODE`: $Url"
    }
}

Write-Host 'Downloading official split manifests and label maps...'
$githubBase = 'https://raw.githubusercontent.com/AI4Bharat/INCLUDE/master'
$metadataFiles = @(
    'train_test_paths/include50_train.txt',
    'train_test_paths/include50_val.txt',
    'train_test_paths/include50_test.txt',
    'train_test_paths/include_train.txt',
    'train_test_paths/include_val.txt',
    'train_test_paths/include_test.txt',
    'label_maps/label_map_include50.json',
    'label_maps/label_map_include.json'
)

foreach ($relativePath in $metadataFiles) {
    $destination = Join-Path $manifestDir ($relativePath -replace '/', '\')
    New-Item -ItemType Directory -Path (Split-Path $destination) -Force | Out-Null
    Invoke-WebRequest -Uri "$githubBase/$relativePath" -OutFile $destination
}

Write-Host 'Reading the official Zenodo v1 file manifest...'
$record = Invoke-RestMethod -Uri "https://zenodo.org/api/records/$recordId"
$archives = @($record.files | Where-Object { $_.key -like '*.zip' } | Sort-Object key)
if ($archives.Count -eq 0) {
    throw 'Zenodo returned no ZIP archives. The record format or availability may have changed.'
}

$archiveNumber = 0
foreach ($file in $archives) {
    $archiveNumber++
    $archivePath = Join-Path $downloadDir $file.key
    $doneMarker = Join-Path $stateDir ($file.key + '.extracted')

    if (Test-Path -LiteralPath $doneMarker) {
        Write-Host "[$archiveNumber/$($archives.Count)] Already extracted: $($file.key)"
        continue
    }

    $driveName = ([System.IO.Path]::GetPathRoot($DataRoot)).TrimEnd('\').TrimEnd(':')
    $freeBytes = (Get-PSDrive -Name $driveName).Free
    $requiredBytes = [int64]$file.size + 3GB
    if ($freeBytes -lt $requiredBytes) {
        throw "Insufficient free space for $($file.key). Need at least $([math]::Round($requiredBytes / 1GB, 1)) GB available."
    }

    Write-Host "[$archiveNumber/$($archives.Count)] Downloading $($file.key)..."
    Invoke-ResumableDownload -Url $file.links.self -Destination $archivePath

    $expectedMd5 = $null
    if ($file.checksum -match '^md5:(.+)$') {
        $expectedMd5 = $Matches[1].ToLowerInvariant()
    }
    if ($expectedMd5) {
        Write-Host "Verifying $($file.key)..."
        $actualMd5 = Get-Md5 $archivePath
        if ($actualMd5 -ne $expectedMd5) {
            throw "Checksum mismatch for $($file.key). Expected $expectedMd5, got $actualMd5."
        }
    }

    Write-Host "Extracting $($file.key)..."
    Expand-Archive -LiteralPath $archivePath -DestinationPath $rawDir -Force
    New-Item -ItemType File -Path $doneMarker -Force | Out-Null

    if (-not $KeepArchives) {
        Remove-Item -LiteralPath $archivePath -Force
    }
}

$videoCount = @(Get-ChildItem -LiteralPath $rawDir -File -Recurse | Where-Object {
    $_.Extension -in @('.mp4', '.avi', '.mov', '.mkv', '.webm')
}).Count

Write-Host "Download and extraction complete. Found $videoCount video files."
Write-Host "Raw videos: $rawDir"
Write-Host "Split manifests: $manifestDir"
Write-Host 'Use the INCLUDE-50 manifests first; the same raw videos can later be used with the full INCLUDE manifests.'

