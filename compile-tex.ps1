param(
    [switch]$CleanOnly
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$buildDir = Join-Path $root "build"
$texFiles = Get-ChildItem -Path $root -Filter "*.tex" -File | Sort-Object Name
$intermediateExtensions = @(
    ".aux", ".bbl", ".bcf", ".blg", ".fdb_latexmk", ".fls", ".lof", ".log",
    ".lot", ".out", ".run.xml", ".synctex.gz", ".toc", ".xdv"
)

function Remove-TeXIntermediateFiles {
    if (Test-Path $buildDir) {
        Remove-Item -LiteralPath $buildDir -Recurse -Force
    }

    Get-ChildItem -Path $root -File | Where-Object {
        $name = $_.Name
        $ext = $_.Extension
        ($intermediateExtensions -contains $ext) -or ($name -like "*.synctex.gz") -or ($name -like "*.run.xml")
    } | Remove-Item -Force
}

Remove-TeXIntermediateFiles

if ($CleanOnly) {
    Write-Host "Cleaned TeX intermediate files."
    exit 0
}

if (-not $texFiles) {
    Write-Host "No .tex files found."
    exit 0
}

foreach ($tex in $texFiles) {
    Write-Host "Compiling $($tex.Name)..."
    latexmk $tex.FullName

    $pdfInBuild = Join-Path $buildDir ($tex.BaseName + ".pdf")
    if (-not (Test-Path $pdfInBuild)) {
        throw "Expected PDF was not created: $pdfInBuild"
    }

    Copy-Item -LiteralPath $pdfInBuild -Destination (Join-Path $root ($tex.BaseName + ".pdf")) -Force
}

Remove-TeXIntermediateFiles
Write-Host "Compiled $($texFiles.Count) TeX file(s). PDFs are in $root"
