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
