# Docker Installation Verification Script

Write-Host "Verifying Docker installation..." -ForegroundColor Green

# Check Docker version
Write-Host "`nChecking Docker version:" -ForegroundColor Yellow
try {
    $dockerVersion = docker --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host $dockerVersion -ForegroundColor Green
    } else {
        throw "Docker command execution failed"
    }
} catch {
    Write-Host "Docker is not installed or not properly configured" -ForegroundColor Red
    Write-Host "Please install Docker Desktop first" -ForegroundColor Yellow
    exit 1
}

# Check Docker Compose version
Write-Host "`nChecking Docker Compose version:" -ForegroundColor Yellow
try {
    $composeVersion = docker-compose --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host $composeVersion -ForegroundColor Green
    } else {
        Write-Host "Docker Compose is not installed" -ForegroundColor Red
    }
} catch {
    Write-Host "Docker Compose is not installed" -ForegroundColor Red
}

# Run test container
Write-Host "`nRunning test container:" -ForegroundColor Yellow
try {
    $testResult = docker run --rm hello-world 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Docker installation successful!" -ForegroundColor Green
        Write-Host $testResult -ForegroundColor Cyan
    } else {
        throw "Test container execution failed"
    }
} catch {
    Write-Host "Docker test failed, please check installation" -ForegroundColor Red
    Write-Host "Make sure Docker Desktop is running" -ForegroundColor Yellow
}

# Check Docker service status
Write-Host "`nChecking Docker service status:" -ForegroundColor Yellow
try {
    $dockerInfo = docker info 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Docker service is running normally" -ForegroundColor Green
        # Display some key information
        $dockerInfo | Select-String -Pattern "Server Version|Operating System|Kernel Version" | ForEach-Object {
            Write-Host $_.ToString() -ForegroundColor Cyan
        }
    } else {
        throw "Cannot get Docker information"
    }
} catch {
    Write-Host "Cannot get Docker information" -ForegroundColor Red
    Write-Host "Please make sure Docker Desktop is running" -ForegroundColor Yellow
}

Write-Host "`nVerification completed!" -ForegroundColor Green 