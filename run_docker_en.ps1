# Run YOLOv11-CFruit Docker Container Script

Write-Host "Starting YOLOv11-CFruit Docker environment..." -ForegroundColor Green

# Check if we're in the project root directory
if (-not (Test-Path "Dockerfile")) {
    Write-Host "Error: Please run this script from the project root directory" -ForegroundColor Red
    exit 1
}

# Create necessary directories
Write-Host "Creating necessary directories..." -ForegroundColor Yellow
$directories = @("data", "checkpoints", "output")
foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        try {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
            Write-Host "Created directory: $dir" -ForegroundColor Cyan
        } catch {
            Write-Host "Failed to create directory: $dir" -ForegroundColor Red
        }
    } else {
        Write-Host "Directory already exists: $dir" -ForegroundColor Gray
    }
}

# Check if Docker is running
Write-Host "Checking Docker service status..." -ForegroundColor Yellow
try {
    $dockerInfo = docker info 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Docker service is not running. Please start Docker Desktop first." -ForegroundColor Red
        exit 1
    }
    Write-Host "Docker service is running normally" -ForegroundColor Green
} catch {
    Write-Host "Cannot connect to Docker service" -ForegroundColor Red
    exit 1
}

# Choose running mode
Write-Host "`nChoose running mode:" -ForegroundColor Yellow
Write-Host "1. CPU version (recommended for testing)" -ForegroundColor White
Write-Host "2. GPU version (requires NVIDIA GPU and nvidia-docker)" -ForegroundColor White

$choice = Read-Host "Please enter your choice (1 or 2)"

if ($choice -eq "2") {
    # GPU version
    Write-Host "Starting GPU version container..." -ForegroundColor Yellow
    try {
        docker-compose up -d yolov11-cfruit-gpu
        if ($LASTEXITCODE -eq 0) {
            Write-Host "GPU container started successfully!" -ForegroundColor Green
            # Enter container
            Write-Host "Entering GPU container..." -ForegroundColor Yellow
            docker exec -it yolov11-cfruit-gpu bash
        } else {
            Write-Host "Failed to start GPU container" -ForegroundColor Red
        }
    } catch {
        Write-Host "Error occurred while starting GPU container" -ForegroundColor Red
    }
} else {
    # CPU version
    Write-Host "Starting CPU version container..." -ForegroundColor Yellow
    try {
        docker-compose up -d yolov11-cfruit
        if ($LASTEXITCODE -eq 0) {
            Write-Host "CPU container started successfully!" -ForegroundColor Green
            # Enter container
            Write-Host "Entering CPU container..." -ForegroundColor Yellow
            docker exec -it yolov11-cfruit bash
        } else {
            Write-Host "Failed to start CPU container" -ForegroundColor Red
        }
    } catch {
        Write-Host "Error occurred while starting CPU container" -ForegroundColor Red
    }
}

Write-Host "`nContainer has been started!" -ForegroundColor Green
Write-Host "Inside the container, you can run the following commands:" -ForegroundColor Cyan
Write-Host "  python test_project.py          # Test project" -ForegroundColor White
Write-Host "  python scripts/prepare_data.py  # Prepare data" -ForegroundColor White
Write-Host "  python scripts/train.py         # Train model" -ForegroundColor White 