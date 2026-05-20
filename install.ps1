# OpenCode DA-Orchestrator — Windows Installer
# Copies skills, prompts, and merges agent config into your opencode.json

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Detect OpenCode config directory
if ($env:OP_ENCODE_HOME) {
    $oconf = $env:OP_ENCODE_HOME
} else {
    $oconf = "$env:USERPROFILE\.config\opencode"
}

$oconfJson = Join-Path $oconf "opencode.json"
$skillsDir = Join-Path $oconf "skills"
$promptsDir = Join-Path $oconf "prompts\da"

Write-Host "🔧 OpenCode DA-Orchestrator Installer" -ForegroundColor Cyan
Write-Host "   Config dir: $oconf" -ForegroundColor Gray

# Step 1: Copy skills
Write-Host "`n📦 Copying DA skills..." -ForegroundColor Yellow
$daSkills = @("da-eda", "da-cleaning", "da-features", "da-modeling", "da-evaluation", "da-interpreter")
foreach ($skill in $daSkills) {
    $src = Join-Path $scriptDir "skills\$skill"
    $dst = Join-Path $skillsDir $skill
    if (-not (Test-Path $src)) {
        Write-Host "   ⚠️  Missing: $src — skipping" -ForegroundColor Red
        continue
    }
    New-Item -ItemType Directory -Force -Path $dst | Out-Null
    Copy-Item "$src\*" $dst -Recurse -Force
    Write-Host "   ✅ $skill" -ForegroundColor Green
}

# Step 2: Copy prompts
Write-Host "`n📝 Copying DA prompts..." -ForegroundColor Yellow
$srcPrompts = Join-Path $scriptDir "prompts\da"
if (Test-Path $srcPrompts) {
    New-Item -ItemType Directory -Force -Path $promptsDir | Out-Null
    Copy-Item "$srcPrompts\*" $promptsDir -Force
    Write-Host "   ✅ prompts/da/" -ForegroundColor Green
} else {
    Write-Host "   ⚠️  No prompts directory found" -ForegroundColor Red
}

# Step 3: Merge agent config into opencode.json
Write-Host "`n⚙️  Merging agent config into opencode.json..." -ForegroundColor Yellow

if (-not (Test-Path $oconfJson)) {
    Write-Host "   ❌ opencode.json not found at $oconfJson" -ForegroundColor Red
    Write-Host "   Create one first, then re-run this script." -ForegroundColor Gray
    exit 1
}

# Read both JSON files
$current = Get-Content $oconfJson -Raw | ConvertFrom-Json
$partialPath = Join-Path $scriptDir "opencode.partial.json"
$partial = Get-Content $partialPath -Raw

# Replace placeholder with actual config home path (backslashes escaped for JSON)
$escapedPath = $oconf -replace '\\', '\\'
$partial = $partial -replace '__OP_ENCODE_HOME__', $escapedPath
$partialObj = $partial | ConvertFrom-Json

# Ensure .agent property exists
if (-not $current.PSObject.Properties['agent']) {
    $current | Add-Member -MemberType NoteProperty -Name 'agent' -Value @{}
}

# Merge agents (skip if already exists)
$added = 0
$skipped = 0
foreach ($prop in $partialObj.agent.PSObject.Properties) {
    $agentName = $prop.Name
    if ($current.agent.PSObject.Properties[$agentName]) {
        Write-Host "   ⏭️  $agentName already exists — skipping" -ForegroundColor Gray
        $skipped++
    } else {
        $current.agent | Add-Member -MemberType NoteProperty -Name $agentName -Value $prop.Value
        Write-Host "   ✅ $agentName" -ForegroundColor Green
        $added++
    }
}

# Write back
$current | ConvertTo-Json -Depth 10 | Set-Content $oconfJson -Encoding UTF8

Write-Host "`n🎉 Done! Added $added agents, skipped $skipped (already existed)." -ForegroundColor Cyan
Write-Host "   Restart OpenCode to load the DA-Orchestrator." -ForegroundColor Gray
Write-Host "   Try: /da-start with a CSV file!" -ForegroundColor Gray
