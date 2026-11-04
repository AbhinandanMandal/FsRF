
# ============================================================
# FsRF - Individual File Commit & Push Script
# One commit per changed/new file
# ============================================================

$ErrorActionPreference = "Stop"

# Verify repository
if (-not (Test-Path ".git")) {
    Write-Host "Error: Run this script from the Git repository root." -ForegroundColor Red
    exit 1
}

# Verify remote
$remote = git remote get-url origin 2>$null

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Git remote 'origin' is not configured." -ForegroundColor Red
    exit 1
}

# Verify branch
$branch = git branch --show-current

if (-not $branch) {
    Write-Host "Error: No active Git branch." -ForegroundColor Red
    exit 1
}

# Avoid accidentally mixing previously staged changes
git diff --cached --quiet

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: You have staged changes. Unstage them before running." -ForegroundColor Red
    exit 1
}

# Collect all pending files
$files = @(
    git ls-files --others --exclude-standard
    git diff --name-only
    git ls-files --deleted
) | Where-Object { $_ -and $_.Trim() } | Sort-Object -Unique

if ($files.Count -eq 0) {
    Write-Host "No pending files to commit." -ForegroundColor Yellow
    exit 0
}

$total = $files.Count
$counter = 0

Write-Host "`nFsRF Individual Commit Process" -ForegroundColor Cyan
Write-Host "Repository : $remote"
Write-Host "Branch     : $branch"
Write-Host "Files      : $total"
Write-Host "Commits    : $total`n"

# Commit every file separately
foreach ($file in $files) {

    $counter++

    $directory = Split-Path $file -Parent
    $filename = Split-Path $file -Leaf

    # Generate a descriptive commit message
    if ($directory) {
        $message = "Update $filename in $directory"
    }
    else {
        $message = "Add or update $filename"
    }

    Write-Host "[$counter/$total] $message" -ForegroundColor Yellow

    # Stage only this file
    git add -A -- "$file"

    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to stage: $file" -ForegroundColor Red
        exit 1
    }

    # Commit this file independently
    git commit -m "$message"

    if ($LASTEXITCODE -ne 0) {
        Write-Host "Commit failed: $file" -ForegroundColor Red
        exit 1
    }
}

# Push all commits together
Write-Host "`nAll $total files committed successfully." -ForegroundColor Green

Write-Host "`nPushing commits to GitHub..." -ForegroundColor Cyan

git push -u origin $branch

if ($LASTEXITCODE -ne 0) {
    Write-Host "Push failed. Your commits remain locally available." -ForegroundColor Red
    exit 1
}

Write-Host "`n============================================" -ForegroundColor Green
Write-Host " PUSH COMPLETED SUCCESSFULLY" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green

Write-Host "Files processed : $total"
Write-Host "Commits created : $total"
Write-Host "Branch          : $branch"
Write-Host "`nLatest commit history:"

git log -$total --oneline
