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
  # Choose an EV directory that contains the EVN/EVA files
  $evDir = $null
  $candidateDirs = $grp.Group | ForEach-Object { $_.Directory.FullName } | Select-Object -Unique
  foreach ($dir in $candidateDirs) {
    $evMatches = Get-ChildItem -Path $dir -Filter ("{0}*.EV?" -f $year) -File -ErrorAction SilentlyContinue
    if ($null -ne $evMatches -and $evMatches.Count -gt 0) { $evDir = $dir; break }
  }
  if (-not $evDir) {
    Write-Host "ERROR: No EVN/EVA files found for $year under expected directories." -ForegroundColor Red
    continue
  }

  # Locate TEAM$year (roster) anywhere under raw root; copy alongside EV files if needed
  $teamFile = Get-ChildItem -Path $rawRoot -Filter ("TEAM{0}" -f $year) -File -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
  $copiedTeam = $false
  if (-not $teamFile) {
    Write-Host "ERROR: TEAM$year not found anywhere under $rawRoot. Extract Retrosheet rosters." -ForegroundColor Red
    continue
  }
  $teamAtEv = Join-Path $evDir ("TEAM{0}" -f $year)
  if (-not (Test-Path -LiteralPath $teamAtEv)) {
    try {
      Copy-Item -LiteralPath $teamFile.FullName -Destination $teamAtEv -Force
      $copiedTeam = $true
    } catch {
      Write-Host ("ERROR: Failed to place TEAM{0} next to EVN/EVA in {1}: {2}" -f $year, $evDir, $_) -ForegroundColor Red
      continue
    }
  }

  # Build season output path
  $seasonOut = Join-Path $seasonOutDir ("retrosheet_events_{0}.csv" -f $year)
  Write-Host "Converting $year from $evDir -> $seasonOut" -ForegroundColor Cyan

  # cd into the EV directory (TEAMYYYY is ensured to exist here now)
  Set-Location $evDir

  # Run cwevent with relative names (so it sees TEAMYYYY)
  $pattern = "${year}*.EV?"
  $matches = Get-ChildItem -Path . -Filter $pattern -File -ErrorAction SilentlyContinue
  if ($null -eq $matches -or $matches.Count -eq 0) {
    Write-Host "ERROR: No files matched pattern '$pattern' in $yearDir. Ensure EVN/EVA files are here." -ForegroundColor Red
    Set-Location $startDir
    continue
  }
  # Try multiple cwevent invocations for compatibility across builds
  # Try 1: no-space -f (include field names with -n). Do NOT pass -x (requires a list).
  & $CwEvent -n -y $year -f0-63 $pattern 2>$null | Out-File -Encoding UTF8 -FilePath $seasonOut

  $ok = (Test-Path $seasonOut) -and ((Get-Item $seasonOut).Length -gt 0)
  if (-not $ok) {
    # Try 2: space -f
    & $CwEvent -n -y $year -f 0-63 $pattern 2>$null | Out-File -Encoding UTF8 -FilePath $seasonOut
    $ok = (Test-Path $seasonOut) -and ((Get-Item $seasonOut).Length -gt 0)
  }
  if (-not $ok) {
    # Try 3: omit -f (default field list)
    & $CwEvent -n -y $year $pattern 2>$null | Out-File -Encoding UTF8 -FilePath $seasonOut
    $ok = (Test-Path $seasonOut) -and ((Get-Item $seasonOut).Length -gt 0)
  }

  if (-not $ok) {
    Write-Host "cwevent failed for $year. Ensure TEAM$year and ${year}*.EV?/EVA are in the same folder. Manual test: cwevent -x -y $year ${year}*.EV? > out.csv" -ForegroundColor Red
    # Clean up copied TEAM file if we created it
    if ($copiedTeam -and (Test-Path -LiteralPath $teamAtEv)) { Remove-Item -LiteralPath $teamAtEv -Force }
    Set-Location $startDir
    exit 1
  }

  # Clean up copied TEAM file to leave tree unchanged
  if ($copiedTeam -and (Test-Path -LiteralPath $teamAtEv)) { Remove-Item -LiteralPath $teamAtEv -Force }
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
