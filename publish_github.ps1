$ErrorActionPreference = "Stop"

$Repo = "ybfo/RockJointNet"
$Description = "Benchmark, figure-generation, and manuscript artifacts for RockJointNet and related rock-joint shear-strength experiments."

if (-not $env:GITHUB_TOKEN) {
    throw "Please set GITHUB_TOKEN to a GitHub PAT with repo scope before running this script."
}

$env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")

$env:GITHUB_TOKEN | gh auth login --hostname github.com --git-protocol https --with-token
gh auth status

git remote set-url origin "https://github.com/$Repo.git"

$exists = $true
try {
    gh repo view $Repo | Out-Null
} catch {
    $exists = $false
}

if (-not $exists) {
    gh repo create $Repo --public --description $Description --source . --remote origin --push
} else {
    git push -u origin main
}

Write-Host "Pushed to https://github.com/$Repo"
