param(
  [string]$CwEvent = "cwevent"
)

# Resolve repo root from script location to build absolute paths
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = Split-Path -Parent $ScriptDir

$rawRoot = Join-Path $RepoRoot "data/raw/retrosheet"
$outRoot = Join-Path $RepoRoot "data/processed"
$seasonOutDir = Join-Path $outRoot "retrosheet_by_season"
New-Item -ItemType Directory -Force -Path $outRoot, $seasonOutDir | Out-Null

# Grab all EVN/EVA files (any nested layout is fine)
$evFiles = Get-ChildItem -Path $rawRoot -Include *.EVN,*.EVA -File -Recurse
if (-not $evFiles) {
  Write-Host "No .EVN/.EVA files found under $rawRoot" -ForegroundColor Yellow
  exit 0
}

# Group by year from the filename prefix (YYYYTTT.EV?)
$byYear = $evFiles | Group-Object { $_.BaseName.Substring(0,4) }

$startDir = Get-Location
foreach ($grp in $byYear) {
  $year = $grp.Name
  # Find the common parent that contains TEAMYYYY (or bail)
  $parents = $grp.Group | ForEach-Object { $_.Directory } | Select-Object -Unique
  $yearDir = $null
  foreach ($p in $parents) {
    if (Test-Path (Join-Path $p.FullName "TEAM$year")) { $yearDir = $p.FullName; break }
  }
  if (-not $yearDir) {
    Write-Host "ERROR: TEAM$year not found in any parent dir of the $year EV files. Extract the Retrosheet ZIP so TEAM$year sits next to  ${year}*.EV? files." -ForegroundColor Red
    continue
  }

  # Build season output path
  $seasonOut = Join-Path $seasonOutDir ("retrosheet_events_{0}.csv" -f $year)
  Write-Host "Converting $year from $yearDir -> $seasonOut" -ForegroundColor Cyan

  # cd into the directory that has TEAMYYYY + EV? files
  Set-Location $yearDir

  # Run cwevent with relative names (so it sees TEAMYYYY)
  $pattern = "${year}*.EV?"
  & $CwEvent -x -y $year -f 0-96 $pattern 2>$null | Out-File -Encoding UTF8 -FilePath $seasonOut

  if ($LASTEXITCODE -ne 0 -or -not (Test-Path $seasonOut)) {
    Write-Host "cwevent failed for $year. Verify cwevent works here: `"$CwEvent`" -V and that TEAM$year is present." -ForegroundColor Red
    Set-Location $startDir
    exit 1
  }
}

# Back to repo root
Set-Location $startDir

# Combine season CSVs into a single CSV with one header
$combined = Join-Path $outRoot "retrosheet_events.csv"
if (Test-Path $combined) { Remove-Item $combined }
$first = $true
Get-ChildItem $seasonOutDir -Filter "retrosheet_events_*.csv" | Sort-Object Name | ForEach-Object {
  $lines = Get-Content $_.FullName
  if ($first) { $lines | Add-Content $combined; $first = $false }
  else { $lines | Select-Object -Skip 1 | Add-Content $combined }
}
Write-Host "Done. Combined CSV: $combined" -ForegroundColor Green
