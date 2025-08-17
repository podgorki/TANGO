# Docker Setup for TANGO

## Prerequisites

- Docker with NVIDIA Container Toolkit installed
- NVIDIA GPU drivers

## Quick Start

1. **Build and run the container:**
   ```bash
   docker-compose up --build
   ```

2. **Run in detached mode:**
   ```bash
   docker-compose up -d
   ```

## Volume Mappings

The following directories are mapped from host to container:

- `./logs` → `/app/logs` - Application logs and debug output
- `./outputs` → `/app/outputs` - Visualization outputs, videos, plots
- `./data` → `/app/data` - Input data and datasets
- `./third_party/models` → `/app/third_party/models` - Model weights
- `./configs` → `/app/configs` - Configuration files (read-only)

## Directory Structure

Ensure these directories exist on your host:
```
TANGO/
├── logs/              # Log files will be written here
├── outputs/           # Visualization outputs (videos, plots)
├── data/              # Input datasets
├── third_party/
│   └── models/        # Model weights (FastSAM, Depth Anything)
└── configs/           # Configuration files
```

## Usage Examples

**Run with custom config:**
```bash
docker-compose run --rm tango python scripts/run_goal_control_demo.py --config_file configs/custom.yaml
```

**Interactive shell:**
```bash
docker-compose run --rm tango bash
```

**View logs:**
```bash
docker-compose logs -f tango
```

## GPU Support

The container uses NVIDIA runtime and requires:
- `nvidia-docker2` or Docker with NVIDIA Container Toolkit
- Compatible NVIDIA drivers on host system

## Troubleshooting

- If GPU is not detected, ensure NVIDIA Container Toolkit is properly installed
- Check that model weights exist in `./third_party/models/` directory
- Verify input data is present in `./data/` directory