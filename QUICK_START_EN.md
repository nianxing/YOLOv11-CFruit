# Quick Start Guide - Windows Docker Setup

## Problem
The original Chinese scripts have encoding issues in PowerShell. Use the English versions instead.

## Solution
Use the English version scripts (files ending with `_en.ps1`):

### Step 1: Fix Execution Policy
```powershell
# Run as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
```

### Step 2: Setup Windows Features
```powershell
# Run as Administrator
.\setup_docker_windows_en.ps1
```

### Step 3: Install Docker Desktop
1. Download Docker Desktop from [Docker website](https://www.docker.com/products/docker-desktop/)
2. Install and restart your computer
3. Start Docker Desktop

### Step 4: Verify Docker Installation
```powershell
.\verify_docker_en.ps1
```

### Step 5: Run the Project
```powershell
.\run_docker_en.ps1
```

## Available Scripts

| Script | Description |
|--------|-------------|
| `setup_docker_windows_en.ps1` | Enable Windows features for Docker |
| `verify_docker_en.ps1` | Verify Docker installation |
| `run_docker_en.ps1` | Start YOLOv11-CFruit containers |
| `fix_execution_policy.ps1` | Fix PowerShell execution policy |

## Manual Commands
If scripts still don't work, use manual commands from `manual_setup_commands.txt`.

## Troubleshooting
- **Encoding issues**: Use English scripts (`_en.ps1`)
- **Execution policy**: Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force`
- **Admin rights**: Run PowerShell as Administrator
- **Docker not running**: Start Docker Desktop first 