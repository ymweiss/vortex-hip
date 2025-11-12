# HIP Metadata Generation - Implementation Checklist

**Goal:** Implement automatic metadata generation in Vortex LLVM compiler
**Approach:** Clang Plugin (Phase 2)
**Timeline:** 6-8 weeks

---

## Prerequisites

- [ ] Vortex LLVM source code available
- [ ] LLVM build environment configured
- [ ] Vortex HIP runtime built and tested
- [ ] Manual metadata test passing (`vecadd_metadata_test`)

**Verification:**
```bash
# Check LLVM source
ls $VORTEX_LLVM_SRC/llvm-project/clang/

# Check build works
cd $VORTEX_LLVM_BUILD && ninja clang

# Check runtime
cd ~/vortex_hip/tests/vecadd_metadata_test && ./vecadd_test
```

---

## Phase 2a: Basic Plugin (Week 1-2)

### Task 1: Create Plugin Skeleton

- [ ] Create directory structure
  ```bash
  mkdir -p llvm-project/clang/examples/VortexHIPPlugin
  cd llvm-project/clang/examples/VortexHIPPlugin
  ```

- [ ] Create `CMakeLists.txt`
  ```cmake
  add_llvm_library(VortexHIPPlugin MODULE VortexHIPPlugin.cpp
    PLUGIN_TOOL clang)

  target_include_directories(VortexHIPPlugin PRIVATE
    ${CLANG_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS})
  ```

- [ ] Add to parent CMakeLists.txt
  ```cmake
  # In llvm-project/clang/examples/CMakeLists.txt
  add_subdirectory(VortexHIPPlugin)
  ```

- [ ] Create `VortexHIPPlugin.cpp` skeleton
  ```cpp
  #include "clang/AST/AST.h"
  #include "clang/AST/ASTConsumer.h"
  #include "clang/Frontend/CompilerInstance.h"
  #include "clang/Frontend/FrontendPluginRegistry.h"

  using namespace clang;

  namespace {
    class VortexHIPMetadataConsumer : public ASTConsumer {
    public:
      void HandleTranslationUnit(ASTContext &Context) override {
        llvm::outs() << "VortexHIP Plugin loaded!\n";
      }
    };

    class VortexHIPAction : public PluginASTAction {
    protected:
      std::unique_ptr<ASTConsumer> CreateASTConsumer(
          CompilerInstance &CI, StringRef InFile) override {
        return std::make_unique<VortexHIPMetadataConsumer>();
      }

      bool ParseArgs(const CompilerInstance &CI,
                     const std::vector<std::string> &args) override {
        return true;
      }
    };
  }

  static FrontendPluginRegistry::Add<VortexHIPAction>
  X("vortex-hip", "Generate HIP kernel metadata");
  ```

- [ ] Build plugin
  ```bash
  cd $VORTEX_LLVM_BUILD
  ninja VortexHIPPlugin
  ```

- [ ] Test plugin loads
  ```bash
  clang++ -fplugin=./lib/VortexHIPPlugin.so \
          -fsyntax-only test.cpp
  # Should print: "VortexHIP Plugin loaded!"
  ```

### Task 2: Implement Kernel Detection

- [ ] Add AST traversal
  ```cpp
  class VortexHIPMetadataConsumer : public ASTConsumer {
    void HandleTranslationUnit(ASTContext &Context) override {
      for (Decl *D : Context.getTranslationUnitDecl()->decls()) {
        if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
          if (isHIPKernel(FD)) {
            llvm::outs() << "Found kernel: " << FD->getName() << "\n";
          }
        }
      }
    }

    bool isHIPKernel(FunctionDecl *FD) {
      // Check for __global__ attribute
      return FD->hasAttr<CUDAGlobalAttr>();
    }
  };
  ```

- [ ] Test with simple kernel
  ```cpp
  __attribute__((global)) void testKernel(int* data) { }
  ```

- [ ] Verify detection works

### Task 3: Extract Basic Type Information

- [ ] Implement type extraction for primitives
  ```cpp
  struct ArgMetadata {
    std::string name;
    uint64_t size;
    uint64_t alignment;
    bool is_pointer;
  };

  std::vector<ArgMetadata> extractArgs(FunctionDecl *FD, ASTContext &Ctx) {
    std::vector<ArgMetadata> args;

    for (ParmVarDecl *Param : FD->parameters()) {
      QualType Type = Param->getType();

      ArgMetadata meta;
      meta.name = Param->getName().str();
      meta.size = Ctx.getTypeSize(Type) / 8;
      meta.alignment = Ctx.getTypeAlign(Type) / 8;
      meta.is_pointer = Type->isPointerType();

      args.push_back(meta);
    }

    return args;
  }
  ```

- [ ] Test with: `void kernel(int* a, float* b, int n)`
- [ ] Verify correct sizes (RV32: ptr=4, int=4, float=4)
- [ ] Verify correct sizes (RV64: ptr=8, int=4, float=4)

### Task 4: Generate Metadata Output

- [ ] Implement code generator
  ```cpp
  void generateMetadata(const std::string &KernelName,
                        const std::vector<ArgMetadata> &Args,
                        raw_ostream &OS) {
    OS << "// Auto-generated for " << KernelName << "\n";
    OS << "static const hipKernelArgumentMetadata "
       << KernelName << "_metadata[] = {\n";

    uint64_t offset = 0;
    for (const auto &arg : Args) {
      // Align offset
      uint64_t padding = (arg.alignment - (offset % arg.alignment)) % arg.alignment;
      offset += padding;

      OS << "  {.offset = " << offset << ", "
         << ".size = " << arg.size << ", "
         << ".alignment = " << arg.alignment << ", "
         << ".is_pointer = " << (arg.is_pointer ? 1 : 0) << "},\n";

      offset += arg.size;
    }

    OS << "};\n";
  }
  ```

- [ ] Add output file argument handling
  ```cpp
  bool ParseArgs(const CompilerInstance &CI,
                 const std::vector<std::string> &args) override {
    for (const auto &arg : args) {
      if (arg.find("-output=") == 0) {
        OutputFile = arg.substr(8);
      }
    }
    return true;
  }
  ```

- [ ] Test output generation
  ```bash
  clang++ -fplugin=VortexHIPPlugin.so \
          -Xclang -plugin-arg-vortex-hip -Xclang -output=metadata.cpp \
          kernel.cpp
  cat metadata.cpp  # Verify output
  ```

**Milestone 1 Complete:** Basic plugin generates metadata for simple kernels

---

## Phase 2b: Full Type System (Week 3-4)

### Task 5: Handle Struct Types

- [ ] Implement struct-by-value handling
  ```cpp
  if (Type->isStructureOrClassType()) {
    const RecordType *RT = Type->getAs<RecordType>();
    const RecordDecl *RD = RT->getDecl();
    const ASTRecordLayout &Layout = Ctx.getASTRecordLayout(RD);

    meta.size = Layout.getSize().getQuantity();
    meta.alignment = Layout.getAlignment().getQuantity();
    meta.is_pointer = false;
  }
  ```

- [ ] Test with: `struct Vec3 { float x, y, z; };`
- [ ] Verify 12-byte size, 4-byte alignment

### Task 6: Handle Arrays

- [ ] Implement array type handling
  ```cpp
  if (Type->isArrayType()) {
    const ConstantArrayType *CAT = Ctx.getAsConstantArrayType(Type);
    QualType ElemType = CAT->getElementType();
    uint64_t NumElems = CAT->getSize().getZExtValue();

    meta.size = (Ctx.getTypeSize(ElemType) / 8) * NumElems;
    meta.alignment = Ctx.getTypeAlign(ElemType) / 8;
  }
  ```

- [ ] Test with: `int data[16]`

### Task 7: Handle Typedefs

- [ ] Add typedef resolution
  ```cpp
  QualType resolveType(QualType Type) {
    while (const TypedefType *TT = Type->getAs<TypedefType>()) {
      Type = TT->getDecl()->getUnderlyingType();
    }
    return Type.getCanonicalType();
  }
  ```

- [ ] Test with: `typedef float Real;`

### Task 8: Architecture Detection

- [ ] Add target info queries
  ```cpp
  const TargetInfo &TI = Context.getTargetInfo();
  unsigned PointerWidth = TI.getPointerWidth(0);
  bool IsRV32 = (PointerWidth == 32);
  ```

- [ ] Test compilation for RV32 and RV64
- [ ] Verify pointer sizes correct

**Milestone 2 Complete:** Plugin handles all C/C++ types correctly

---

## Phase 2c: Integration (Week 5-6)

### Task 9: Generate Complete Registration Code

- [ ] Add binary symbol references
  ```cpp
  OS << "extern \"C\" {\n";
  OS << "  extern const uint8_t kernel_vxbin[];\n";
  OS << "  extern const uint8_t kernel_vxbin_end[];\n";
  OS << "}\n\n";
  ```

- [ ] Add kernel handle declaration
  ```cpp
  OS << "void* " << KernelName << "_handle = nullptr;\n\n";
  ```

- [ ] Add registration function
  ```cpp
  OS << "__attribute__((constructor))\n";
  OS << "static void register_" << KernelName << "() {\n";
  OS << "  __hipRegisterFunctionWithMetadata(...);\n";
  OS << "}\n";
  ```

- [ ] Test with vecadd kernel
- [ ] Compare output to manual `kernel_metadata_manual.cpp`

### Task 10: Build System Integration

- [ ] Install plugin with LLVM
  ```cmake
  install(TARGETS VortexHIPPlugin
          DESTINATION ${LLVM_LIBRARY_DIR}/plugins)
  ```

- [ ] Create wrapper script `hip-clang++`
  ```bash
  #!/bin/bash
  PLUGIN_PATH="$(dirname $0)/../lib/plugins/VortexHIPPlugin.so"
  clang++ -fplugin=$PLUGIN_PATH \
          -Xclang -plugin-arg-vortex-hip \
          -Xclang -output=${OUTPUT_FILE} \
          "$@"
  ```

- [ ] Add to CMake find script
  ```cmake
  # FindVortexHIP.cmake
  find_library(VORTEX_HIP_PLUGIN VortexHIPPlugin
               PATHS ${LLVM_LIBRARY_DIR}/plugins)
  ```

- [ ] Update vecadd_metadata_test Makefile
  ```makefile
  kernel_metadata.cpp: kernel.cpp
  	hip-clang++ -fsyntax-only \
  	            -Xclang -plugin-arg-vortex-hip \
  	            -Xclang -output=$@ \
  	            $<
  ```

- [ ] Test automated build
  ```bash
  cd ~/vortex_hip/tests/vecadd_metadata_test
  make clean && make
  ```

### Task 11: Error Handling

- [ ] Add validation for kernel signatures
- [ ] Check for unsupported types
- [ ] Provide helpful error messages
- [ ] Test error cases

**Milestone 3 Complete:** Integrated into build system, automated

---

## Phase 2d: Production (Week 7+)

### Task 12: Comprehensive Testing

- [ ] Create test suite in `vortex_hip/tests/compiler/`
- [ ] Test simple kernels (various types)
- [ ] Test complex kernels (structs, arrays)
- [ ] Test edge cases (empty args, single arg)
- [ ] Test RV32 vs RV64
- [ ] Integration tests with runtime

### Task 13: Documentation

- [ ] User guide for hip-clang++
- [ ] Plugin API documentation
- [ ] Troubleshooting guide
- [ ] Update README files

### Task 14: Optimization

- [ ] Cache type calculations
- [ ] Minimize string operations
- [ ] Profile plugin performance

### Task 15: CI/CD Integration

- [ ] Add to Vortex CI pipeline
- [ ] Automated tests on each commit
- [ ] Regression testing

**Milestone 4 Complete:** Production-ready implementation

---

## Validation Checklist

At each milestone, verify:

- [ ] Compiles without warnings
- [ ] Passes all unit tests
- [ ] Manual test (vecadd) works
- [ ] Documentation updated
- [ ] Code reviewed

---

## Success Criteria

**Phase 2 is complete when:**

✅ Plugin automatically detects HIP kernels
✅ Extracts accurate type information
✅ Generates correct metadata
✅ Generates complete registration code
✅ Integrates with build system
✅ All tests pass
✅ Documented and maintainable

---

## Rollback Plan

If Phase 2 takes too long or encounters blockers:

**Fallback:** Continue using Phase 1 (manual metadata)
- Works today
- Validated
- No risk

**Hybrid Approach:**
- Use plugin for simple cases
- Manual for complex cases
- Gradually expand plugin capabilities

---

## Resources Needed

- [ ] Access to Vortex LLVM source
- [ ] Build machine with LLVM dev environment
- [ ] Time: ~40-60 hours of development
- [ ] Testing infrastructure
- [ ] Code review from Vortex team

---

## Contact / Questions

- **LLVM Plugin Questions:** Refer to Clang plugin documentation
- **Vortex Specifics:** Check Vortex LLVM GitHub issues
- **HIP Runtime:** See `vortex_hip/runtime/ARGUMENT_MARSHALING.md`
- **This Checklist:** Keep updated as implementation progresses

---

**Created:** 2025-11-06
**Status:** Ready to Start
**Next Action:** Begin Task 1 - Create Plugin Skeleton
