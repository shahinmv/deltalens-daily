#!/bin/bash

# Daily tasks runner script
# Executes the scripts in the specified order with error handling and logging

# Set the script directory
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

# Create startup_logs directory if it doesn't exist
mkdir -p startup_logs

# Function to log messages with timestamp
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a startup_logs/daily_run.log
}

# Function to run a script with error handling
run_script() {
    local script_name=$1
    local script_description=$2
    
    log "Starting $script_description..."
    
    if python3 "$script_name"; then
        log "$script_description completed successfully"
        return 0
    else
        local exit_code=$?
        log "ERROR: $script_description failed with exit code $exit_code"
        return $exit_code
    fi
}

# Main execution
log "=========================================="
log "Starting daily data collection and signal generation"
log "=========================================="

# 1. Back fill Open Interest data
if run_script "back_fill_oi.py" "Open Interest data collection"; then
    log "✓ Open Interest data collection completed"
else
    log "✗ Open Interest data collection failed - continuing with next task"
fi

# 2. Back fill Funding Rate data
if run_script "back_fill_funding.py" "Funding Rate data collection"; then
    log "✓ Funding Rate data collection completed"
else
    log "✗ Funding Rate data collection failed - continuing with next task"
fi

# 3. Extract news data
if run_script "news_extract.py" "News data extraction"; then
    log "✓ News data extraction completed"
else
    log "✗ News data extraction failed - continuing with next task"
fi

# 4. Generate trading signal
if run_script "generate_signal.py" "Signal generation"; then
    log "✓ Signal generation completed"
else
    log "✗ Signal generation failed"
    exit 1
fi

# 5. Check trades status (final step)
if run_script "check_trades_status.py" "Trade status checking"; then
    log "✓ Trade status checking completed"
else
    log "✗ Trade status checking failed - continuing as this is not critical"
fi

log "=========================================="
log "Daily tasks completed successfully!"
log "=========================================="