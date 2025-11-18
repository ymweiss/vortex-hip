#!/bin/bash
# Generate MLIR GPU dialect test cases from HIP test suite
# Purpose: Create comprehensive test files for metadata extraction and kernel launch development

set -e

# Configuration
HIP_TESTS_DIR="hip_tests"
OUTPUT_DIR="hip_tests/mlir_output"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VORTEX_HIP_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "==================================================================="
echo "Generating MLIR GPU Dialect Test Suite from HIP Tests"
echo "==================================================================="
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Change to vortex_hip root
cd "$VORTEX_HIP_ROOT"

# Find all HIP test files
HIP_FILES=("$HIP_TESTS_DIR"/*.hip)

if [ ${#HIP_FILES[@]} -eq 0 ]; then
  echo "Error: No .hip files found in $HIP_TESTS_DIR/"
  exit 1
fi

echo "Found ${#HIP_FILES[@]} HIP test files"
echo ""

# Statistics
SUCCESS_COUNT=0
FAIL_COUNT=0
FAILED_FILES=()

# Process each HIP file
for hip_file in "${HIP_FILES[@]}"; do
  # Skip if glob didn't match anything
  if [ ! -f "$hip_file" ]; then
    continue
  fi

  base=$(basename "$hip_file" .hip)
  output_file="${OUTPUT_DIR}/${base}_gpu_dialect.mlir"

  echo "Processing: $base.hip"
  echo "  Input:  $hip_file"
  echo "  Output: $output_file"

  # Run conversion script
  if ./scripts/polygeist/hip-to-gpu-dialect.sh "$hip_file" "$output_file" 2>&1 | grep -q "error:"; then
    echo "  Status: ❌ FAILED"
    FAIL_COUNT=$((FAIL_COUNT + 1))
    FAILED_FILES+=("$base")
  else
    echo "  Status: ✅ SUCCESS"
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))

    # Quick analysis of generated MLIR
    if [ -f "$output_file" ]; then
      # Count gpu.launch_func occurrences
      launch_count=$(grep -c "gpu.launch_func" "$output_file" 2>/dev/null || echo "0")
      # Count gpu.barrier occurrences
      barrier_count=$(grep -c "gpu.barrier" "$output_file" 2>/dev/null || echo "0")

      echo "    Launch functions: $launch_count"
      echo "    Barriers: $barrier_count"

      # Extract argument types from first launch_func
      if [ "$launch_count" -gt 0 ]; then
        args_line=$(grep -A 2 "gpu.launch_func" "$output_file" | grep "args(" | head -1)
        echo "    Arguments: $args_line"
      fi
    fi
  fi

  echo ""
done

# Summary
echo "==================================================================="
echo "Summary"
echo "==================================================================="
echo "Total files processed: $((SUCCESS_COUNT + FAIL_COUNT))"
echo "Successful: $SUCCESS_COUNT"
echo "Failed: $FAIL_COUNT"

if [ $FAIL_COUNT -gt 0 ]; then
  echo ""
  echo "Failed files:"
  for failed in "${FAILED_FILES[@]}"; do
    echo "  - $failed.hip"
  done
fi

echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""

# Generate analysis script
ANALYSIS_SCRIPT="${OUTPUT_DIR}/analyze_launch_patterns.sh"
cat > "$ANALYSIS_SCRIPT" << 'EOF'
#!/bin/bash
# Analyze gpu.launch_func patterns in generated MLIR files

echo "==================================================================="
echo "GPU Launch Function Argument Pattern Analysis"
echo "==================================================================="
echo ""

for mlir_file in hip_tests/mlir_output/*_gpu_dialect.mlir; do
  if [ ! -f "$mlir_file" ]; then
    continue
  fi

  base=$(basename "$mlir_file" _gpu_dialect.mlir)
  echo "--- $base ---"

  # Extract launch_func operations
  grep -A 3 "gpu.launch_func" "$mlir_file" | while IFS= read -r line; do
    if [[ "$line" =~ gpu.launch_func ]]; then
      echo "  Kernel: $line"
    elif [[ "$line" =~ blocks\ in ]]; then
      echo "  Grid:   $line"
    elif [[ "$line" =~ threads\ in ]]; then
      echo "  Block:  $line"
    elif [[ "$line" =~ args\( ]]; then
      echo "  Args:   $line"

      # Parse argument types
      args=$(echo "$line" | sed 's/.*args(\(.*\)).*/\1/')
      echo "    Argument types:"

      # Extract each argument
      IFS=',' read -ra ARG_ARRAY <<< "$args"
      arg_num=0
      for arg in "${ARG_ARRAY[@]}"; do
        # Clean up whitespace
        arg=$(echo "$arg" | xargs)

        # Extract type
        if [[ "$arg" =~ :\ (.*) ]]; then
          type="${BASH_REMATCH[1]}"
          arg_num=$((arg_num + 1))

          # Determine if pointer or value
          if [[ "$type" =~ memref ]]; then
            size="8 bytes (pointer)"
          elif [[ "$type" =~ i64|f64 ]]; then
            size="8 bytes (value)"
          elif [[ "$type" =~ i32|f32 ]]; then
            size="4 bytes (value)"
          elif [[ "$type" =~ i16|f16 ]]; then
            size="2 bytes (value)"
          elif [[ "$type" =~ i8 ]]; then
            size="1 byte (value)"
          else
            size="unknown"
          fi

          echo "      arg$arg_num: $type -> $size"
        fi
      done
    fi
  done

  echo ""
done

echo "==================================================================="
echo "Use this analysis for metadata extraction implementation"
echo "==================================================================="
EOF

chmod +x "$ANALYSIS_SCRIPT"

echo "Generated analysis script: $ANALYSIS_SCRIPT"
echo "Run it with: ./$ANALYSIS_SCRIPT"
echo ""

# Auto-run analysis if all succeeded
if [ $FAIL_COUNT -eq 0 ]; then
  echo "All conversions successful! Running argument pattern analysis..."
  echo ""
  bash "$ANALYSIS_SCRIPT"
fi

echo "==================================================================="
echo "Next Steps:"
echo "==================================================================="
echo "1. Review generated MLIR files in: $OUTPUT_DIR"
echo "2. Run analysis script: ./$ANALYSIS_SCRIPT"
echo "3. Use patterns for metadata extraction implementation"
echo "4. Create FileCheck tests based on these patterns"
echo ""
