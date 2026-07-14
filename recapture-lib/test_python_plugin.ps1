param(
    [Parameter(Mandatory = $true)]
    [string]$Path
)

Write-Host "Testing if path is a directory...              " -NoNewline

if (!(Test-Path -Path $Path -PathType Container)) {
    Write-Host ""
    Write-Host "Error: The provided argument is not a directory." -ForegroundColor Red
    Write-Host "Please provide a directory as the first and only argument. For example:"
    Write-Host ".\test_python_plugin.ps1 `"C:\Users\Max\Documents\Recapture-Plugins\example-organisation\example-plugin`""
    exit 1
} else {
    Write-Host "Done" -ForegroundColor Green
}


Write-Host "Testing if given folder contains a main.py...  " -NoNewline

$entryPoint = Join-Path $Path "main.py"

if (!(Test-Path -Path $entryPoint)) {
    Write-Host ""
    Write-Host "Error: The provided directory does not have a main.py in it." -ForegroundColor Red
    Write-Host "It is required that Recapture plugins have a main.py file, as that is the entry point to the plugin."
    exit 1
} else {
    Write-Host "Done" -ForegroundColor Green
}

Write-Host "Locating pixi...                               " -NoNewline

$pixiFound = $false
$pixiLocation = "pixi" # assume it's on path first

if (!(Get-Command $pixiLocation -ErrorAction SilentlyContinue)) {
    Write-Host ""
    Write-Host "Pixi is not installed on your machine. Falling back to bundled one."
    Write-Host "Warning: the bundled pixi may be out of date. It is best to install it on your machine." -ForegroundColor Yellow
    Write-Host "This can be done by running the following command:"
    Write-Host "winget install prefix-dev.pixi --source winget"
    Write-Host "Or by following the instructions at https://pixi.prefix.dev/latest/installation/."
} else {
    $pixiFound = $true
    Write-Host "Done" -ForegroundColor Green
}

if (!$pixiFound) {
    Write-Host ""
    Write-Host "    Locating bundled pixi...                   " -NoNewline
    $pixiLocation = Join-Path $PSScriptRoot "./pixi-x86_64-pc-windows-msvc.exe"
    if (!(Test-Path -Path $pixiLocation)) {
        Write-Host ""
        Write-Host "    Error: No pixi can be located. Refer to the above warning for how to install it. Please also contact the Impact Lab about this error." -ForegroundColor Red
        exit 1
    } else {
        Write-Host "Done" -ForegroundColor Green
    }
}

Write-Host "Running plugin using pixi"
Write-Host ""

$trialDir = Join-Path $PSScriptRoot "example-session/walk"
$osimPath = Join-Path $PSScriptRoot "example-session/models/LaiUhlrich2022_scaled.osim"
$trcPath = Join-Path $PSScriptRoot "example-session/walk/P06_walkPref.trc"
$motPath = Join-Path $PSScriptRoot "example-session/walk/P06_walkPref.mot"

$outputPath = Join-Path $trialDir "spec.json"

# check and delete existing spec.json
if (Test-Path -Path $outputPath) {
    Write-Host "Deleting old spec file."
    Remove-Item -Path $outputPath
}

$command = "$pixiLocation run -e recapture-opensim python $entryPoint $trialDir $osimPath $trcPath $motPath"
#Write-Host "Exact command: $command"

Invoke-Expression $command
# $pluginOutput = Invoke-Expression $command

# Write-Host ""
# Write-Host "Output of plugin:"
# Write-Host ""
# Write-Host $pluginOutput
# Write-Host ""

If (!(Test-Path -Path $outputPath)) {
    Write-Host ""
    Write-Host "Error: The plugin did not output a spec file in the required location." -ForegroundColor Red
    Write-Host "    The expected location is $outputPath"
    exit 1
}
