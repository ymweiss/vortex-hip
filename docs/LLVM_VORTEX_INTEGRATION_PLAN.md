# llvm-vortex Integration Plan

**Goal:** Integrate Polygeist with llvm-vortex to eliminate duplicate LLVM builds and prepare for mainline Vortex merge.

## Approach: Cherry-pick KernelOutlining Modifications

Cherry-pick the 5 KernelOutlining commits from Polygeist's llvm-project to llvm-vortex.

### Commits to Cherry-pick

| Order | Commit | Description |
|-------|--------|-------------|
| 1 | `ecdd57a09166` | Add kernel_arg_mapping attribute to preserve host-device arg order |
| 2 | `5b992e15462e` | Propagate vortex metadata through kernel outlining |
| 3 | `8fb468b7fd68` | Add semantic inference for synthetic kernel arguments |
| 4 | `b67e4fa41379` | Add blockDimXY semantic detection and dynamic dim3 positions |
| 5 | `2e1b9f607e27` | Remove launch_wrapper skip in KernelOutlining |

### Files Modified

All changes are in a single file:
- `mlir/lib/Dialect/GPU/Transforms/KernelOutlining.cpp`

Additional includes added:
- `mlir/Dialect/Func/IR/FuncOps.h`
- `mlir/Dialect/MemRef/IR/MemRef.h`

---

## Implementation Steps

### Phase 1: Prepare llvm-vortex Branch

```bash
cd /home/yaakov/vortex_hip/llvm-vortex
git checkout -b yaakov/polygeist-kerneloutlining

# Verify base version compatibility
git log --oneline -1  # Should be LLVM 18.x based
```

### Phase 2: Cherry-pick Commits

```bash
# Add Polygeist's llvm-project as remote
git remote add polygeist-llvm /home/yaakov/vortex_hip/Polygeist/llvm-project

# Fetch the commits
git fetch polygeist-llvm yaakov/cgeist-kernel-metadata

# Cherry-pick in order (oldest first)
git cherry-pick ecdd57a09166  # kernel_arg_mapping attribute
git cherry-pick 5b992e15462e  # vortex metadata propagation
git cherry-pick 8fb468b7fd68  # synthetic semantic inference
git cherry-pick b67e4fa41379  # blockDimXY detection
git cherry-pick 2e1b9f607e27  # remove launch_wrapper skip
```

### Phase 3: Resolve Conflicts (if any)

The commits modify LLVM 18 KernelOutlining.cpp. Potential conflicts:
- API differences between LLVM 18 versions (minor vs patch level)
- Upstream changes to KernelOutlining.cpp

Resolution strategy: Keep our modifications, adapt to any API changes.

### Phase 4: Build llvm-vortex with MLIR

```bash
cd /home/yaakov/vortex_hip/llvm-vortex
cmake -G Ninja -S llvm -B build \
  -DLLVM_ENABLE_PROJECTS="clang;mlir" \
  -DLLVM_TARGETS_TO_BUILD="host;RISCV" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON
ninja -C build
```

### Phase 5: Configure Polygeist Against llvm-vortex

```bash
cd /home/yaakov/vortex_hip/Polygeist
cmake -G Ninja -B build-vortex \
  -DMLIR_DIR=/home/yaakov/vortex_hip/llvm-vortex/build/lib/cmake/mlir \
  -DCLANG_DIR=/home/yaakov/vortex_hip/llvm-vortex/build/lib/cmake/clang \
  -DLLVM_DIR=/home/yaakov/vortex_hip/llvm-vortex/build/lib/cmake/llvm \
  -DLLVM_EXTERNAL_LIT=/home/yaakov/vortex_hip/llvm-vortex/build/bin/llvm-lit
ninja -C build-vortex cgeist polygeist-opt
```

### Phase 6: Test HIP Pipeline

```bash
# Test single-kernel
./scripts/compile_hip_v2.sh hip_tests/vecadd.hip -o hip_tests/vecadd_test

# Test multi-kernel
./scripts/compile_hip_v2.sh hip_tests/dogfood.hip -o hip_tests/dogfood_test

# Run on SimX
cd hip_tests
VORTEX_HOME=/home/yaakov/vortex_hip/vortex \
LD_LIBRARY_PATH=/home/yaakov/vortex_hip/vortex/build/runtime:$LD_LIBRARY_PATH \
VORTEX_KERNEL_PATH=./ ./vecadd_test
```

### Phase 7: Remove Polygeist llvm-project Submodule

Once tests pass:
```bash
cd /home/yaakov/vortex_hip/Polygeist
git submodule deinit llvm-project
git rm llvm-project
rm -rf .git/modules/llvm-project
git commit -m "[refactor] Remove llvm-project submodule, use external llvm-vortex"
```

---

## Benefits

| Metric | Before | After |
|--------|--------|-------|
| LLVM builds | 2 (llvm-vortex + Polygeist llvm-project) | 1 (llvm-vortex only) |
| Build time | ~4-6 hours | ~2-3 hours |
| Storage | ~200GB | ~100GB |
| Maintenance | 2 LLVM forks to track | 1 LLVM fork |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Cherry-pick conflicts | Manual resolution, may need to adapt to API differences |
| llvm-vortex updates | Rebase our branch when vortex updates LLVM base |
| cgeist API changes | Test incrementally, fix issues as found |

---

## Success Criteria

1. All 5 commits cherry-picked cleanly to llvm-vortex
2. llvm-vortex builds with MLIR enabled
3. Polygeist builds against llvm-vortex
4. All 23 HIP tests pass (100%)
5. Polygeist llvm-project submodule removed
