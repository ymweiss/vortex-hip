#!/bin/bash

# Monitor Polygeist build progress
# Checks every 2 minutes to ensure build is still running

BUILD_DIR=/home/yaakov/vortex_hip/Polygeist/llvm-project/build
POLYGEIST_BUILD_DIR=/home/yaakov/vortex_hip/Polygeist/build
LOG_FILE=/home/yaakov/vortex_hip/Polygeist/build-monitor.log

echo "=== Build Monitor Started: $(date) ===" | tee -a $LOG_FILE

while true; do
    sleep 120  # Check every 2 minutes

    echo "" | tee -a $LOG_FILE
    echo "--- Check at $(date) ---" | tee -a $LOG_FILE

    # Check if ninja is running
    NINJA_COUNT=$(pgrep -c ninja || true)
    echo "Ninja processes running: $NINJA_COUNT" | tee -a $LOG_FILE

    # Check LLVM build directory
    if [ -d "$BUILD_DIR" ]; then
        LLVM_OBJS=$(find $BUILD_DIR -name "*.o" 2>/dev/null | wc -l)
        LLVM_LIBS=$(find $BUILD_DIR -name "*.a" 2>/dev/null | wc -l)
        echo "LLVM build: $LLVM_OBJS object files, $LLVM_LIBS libraries" | tee -a $LOG_FILE

        # Check for key binaries
        if [ -f "$BUILD_DIR/bin/clang" ]; then
            echo "  ✓ clang built" | tee -a $LOG_FILE
        fi
        if [ -f "$BUILD_DIR/bin/mlir-opt" ]; then
            echo "  ✓ mlir-opt built" | tee -a $LOG_FILE
        fi
    fi

    # Check Polygeist build directory
    if [ -d "$POLYGEIST_BUILD_DIR" ]; then
        POLYGEIST_OBJS=$(find $POLYGEIST_BUILD_DIR -name "*.o" 2>/dev/null | wc -l)
        echo "Polygeist build: $POLYGEIST_OBJS object files" | tee -a $LOG_FILE

        if [ -f "$POLYGEIST_BUILD_DIR/bin/cgeist" ]; then
            echo "  ✓✓✓ cgeist built - BUILD COMPLETE! ✓✓✓" | tee -a $LOG_FILE
            break
        fi
    fi

    # Check system load
    LOAD=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}')
    echo "System load: $LOAD" | tee -a $LOG_FILE

    # If no ninja running and cgeist not built, might be stuck
    if [ $NINJA_COUNT -eq 0 ] && [ ! -f "$POLYGEIST_BUILD_DIR/bin/cgeist" ]; then
        echo "⚠️  WARNING: No ninja processes running but build incomplete!" | tee -a $LOG_FILE
        echo "  Check build logs and consider resuming with: ninja -j8 -C $BUILD_DIR" | tee -a $LOG_FILE
    fi
done

echo "=== Build Monitor Finished: $(date) ===" | tee -a $LOG_FILE
