# Daily Run Docker Setup

This Docker setup containerizes the daily data collection and signal generation process for the Bitcoin trading system.

## Files Created

- `Dockerfile`: Main container configuration
- `requirements.txt`: Python dependencies
- `run_daily_tasks.sh`: Bash script that executes the tasks in order
- `docker-compose.yml`: Docker Compose configuration
- `.dockerignore`: Files to exclude from Docker build context

## Execution Order

The container will execute these scripts in the specified order:

1. **back_fill_oi.py** - Collects Bitcoin open interest data from Binance
2. **back_fill_funding.py** - Collects funding rate data from Binance  
3. **news_extract.py** - Extracts Bitcoin news from Cointelegraph
4. **generate_signal.py** - Generates trading signals using ML models

## How to Use

### Build and Run with Docker Compose (Recommended)

```bash
# Navigate to the daily-run directory
cd research/daily-run

# Build and run the container
docker-compose up --build

# Run in detached mode (background)
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop the container
docker-compose down
```

### Build and Run with Docker

```bash
# Build the image
docker build -t daily-run .

# Run the container
docker run --name daily-run-container daily-run

# Run with volume mounts for logs
docker run --name daily-run-container \
  -v $(pwd)/startup_logs:/app/startup_logs \
  -v $(pwd)/data:/app/data \
  daily-run
```

## Features

- **Error Handling**: Each script continues even if previous ones fail
- **Logging**: All output is logged to `startup_logs/daily_run.log`
- **Volume Mounts**: Logs and data are preserved outside the container
- **Lightweight**: Uses Python 3.9 slim base image
- **Sequential Execution**: Tasks run in the correct dependency order

## Monitoring

Logs are written to both console and `startup_logs/daily_run.log`. You can monitor progress with:

```bash
# If using docker-compose
docker-compose logs -f

# If using docker run
docker logs -f daily-run-container

# Or check the log file
tail -f startup_logs/daily_run.log
```

## Customization

- Modify `run_daily_tasks.sh` to change execution order or add new tasks
- Update `requirements.txt` to add new Python dependencies
- Adjust `docker-compose.yml` for scheduling or environment variables

## Scheduling

For production use, you can:
1. Use cron to run the container daily
2. Uncomment the scheduled loop in `docker-compose.yml`
3. Use Kubernetes CronJobs
4. Use cloud scheduler services (AWS CloudWatch Events, etc.)