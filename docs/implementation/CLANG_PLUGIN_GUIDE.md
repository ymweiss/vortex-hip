# Clang Plugin Development Guide: AST Traversal and Transformation

**Document Version:** 1.0
**Date:** 2025-11-09
**Target:** Phase 2 HIP-to-Vortex Plugin Implementation

---

## Table of Contents

1. [Overview](#overview)
2. [Clang Plugin Architecture](#clang-plugin-architecture)
3. [Building a Basic Plugin](#building-a-basic-plugin)
4. [AST Traversal APIs](#ast-traversal-apis)
5. [Working with Function Declarations](#working-with-function-declarations)
6. [Extracting Type Information](#extracting-type-information)
7. [Transforming AST Nodes](#transforming-ast-nodes)
8. [Generating Output](#generating-output)
9. [Complete Example: HIP Metadata Extractor](#complete-example-hip-metadata-extractor)
10. [Building and Using the Plugin](#building-and-using-the-plugin)
11. [Debugging and Testing](#debugging-and-testing)

---

## Overview

### What is a Clang Plugin?

A Clang plugin is a dynamic library that extends Clang's functionality by hooking into the compilation process. Plugins can:
- Inspect the Abstract Syntax Tree (AST)
- Perform custom analysis
- Generate additional output
- Transform code (with AST rewriting)

### Why Use Plugins for Phase 2?

**Advantages:**
- No need to modify LLVM source
- Can be compiled and loaded separately
- Easier to maintain and distribute
- Direct access to AST with full type information

**Use Case for Phase 2:**
- Extract HIP kernel metadata from AST
- Transform HIP constructs to Vortex equivalents
- Generate registration code

---

## Clang Plugin Architecture

### Plugin Lifecycle

```
Clang Driver
    ↓
[1] Load Plugin (.so file)
    ↓
[2] Register Plugin Actions
    ↓
[3] Parse Source → AST
    ↓
[4] Run Semantic Analysis
    ↓
[5] ⭐ Plugin Executes (AST Consumer)
    │   - Traverse AST
    │   - Extract information
    │   - Transform nodes
    │   - Generate output
    ↓
[6] Continue to CodeGen
```

### Key Plugin Components

```cpp
// 1. Plugin Action (Entry Point)
class MyPluginAction : public PluginASTAction {
  std::unique_ptr<ASTConsumer> CreateASTConsumer(...) override;
  bool ParseArgs(...) override;
};

// 2. AST Consumer (Processes Translation Unit)
class MyASTConsumer : public ASTConsumer {
  void HandleTranslationUnit(ASTContext &Context) override;
};

// 3. AST Visitor (Traverses AST Nodes)
class MyASTVisitor : public RecursiveASTVisitor<MyASTVisitor> {
  bool VisitFunctionDecl(FunctionDecl *FD);
  bool VisitCallExpr(CallExpr *CE);
  // ... other Visit methods
};

// 4. Plugin Registration (Makes plugin discoverable)
static FrontendPluginRegistry::Add<MyPluginAction>
  X("my-plugin", "My plugin description");
```

---

## Building a Basic Plugin

### Minimal Plugin Structure

```cpp
// File: MyPlugin.cpp

#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;

namespace {

// Step 1: Define AST Visitor
class MyVisitor : public RecursiveASTVisitor<MyVisitor> {
private:
  ASTContext *Context;

public:
  explicit MyVisitor(ASTContext *Context) : Context(Context) {}

  // Visit function declarations
  bool VisitFunctionDecl(FunctionDecl *FD) {
    if (FD->hasBody()) {
      llvm::outs() << "Found function: " << FD->getName() << "\n";
    }
    return true;  // Continue traversal
  }
};

// Step 2: Define AST Consumer
class MyConsumer : public ASTConsumer {
private:
  MyVisitor Visitor;

public:
  explicit MyConsumer(ASTContext *Context) : Visitor(Context) {}

  // Called when entire translation unit is parsed
  void HandleTranslationUnit(ASTContext &Context) override {
    // Traverse the entire AST
    Visitor.TraverseDecl(Context.getTranslationUnitDecl());
  }
};

// Step 3: Define Plugin Action
class MyPluginAction : public PluginASTAction {
protected:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(
      CompilerInstance &CI, llvm::StringRef) override {
    return std::make_unique<MyConsumer>(&CI.getASTContext());
  }

  bool ParseArgs(const CompilerInstance &CI,
                 const std::vector<std::string> &args) override {
    // Parse plugin-specific arguments
    for (const auto &arg : args) {
      llvm::errs() << "Plugin arg: " << arg << "\n";
    }
    return true;
  }
};

} // namespace

// Step 4: Register Plugin
static FrontendPluginRegistry::Add<MyPluginAction>
  X("my-plugin", "My example plugin");
```

### Building the Plugin

```bash
# CMakeLists.txt
cmake_minimum_required(VERSION 3.13.4)
project(MyPlugin)

find_package(LLVM REQUIRED CONFIG)
find_package(Clang REQUIRED CONFIG)

add_definitions(${LLVM_DEFINITIONS})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${CLANG_INCLUDE_DIRS})

add_library(MyPlugin MODULE MyPlugin.cpp)
target_link_libraries(MyPlugin PRIVATE
  clangAST
  clangBasic
  clangFrontend
  clangTooling
)

# Build
mkdir build && cd build
cmake -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm ..
make
```

### Using the Plugin

```bash
# Load plugin during compilation
clang++ -fplugin=./MyPlugin.so -c input.cpp

# Pass arguments to plugin
clang++ -fplugin=./MyPlugin.so -plugin-arg-my-plugin arg1 -c input.cpp
```

---

## AST Traversal APIs

### RecursiveASTVisitor

**Purpose:** Automatically traverse all AST nodes

**Key Methods to Override:**

```cpp
class MyVisitor : public RecursiveASTVisitor<MyVisitor> {
public:
  // Visit declarations
  bool VisitDecl(Decl *D);
  bool VisitFunctionDecl(FunctionDecl *FD);
  bool VisitVarDecl(VarDecl *VD);
  bool VisitParmVarDecl(ParmVarDecl *PVD);
  bool VisitRecordDecl(RecordDecl *RD);

  // Visit statements
  bool VisitStmt(Stmt *S);
  bool VisitCallExpr(CallExpr *CE);
  bool VisitDeclRefExpr(DeclRefExpr *DRE);
  bool VisitIfStmt(IfStmt *IS);
  bool VisitForStmt(ForStmt *FS);

  // Visit types
  bool VisitQualType(QualType QT);
  bool VisitPointerType(PointerType *PT);

  // Return true to continue traversal, false to stop
};
```

### Traversal Control

```cpp
// Start traversal from translation unit
Visitor.TraverseDecl(Context.getTranslationUnitDecl());

// Traverse specific node
Visitor.TraverseStmt(someStatement);

// Manual traversal (more control)
for (Decl *D : Context.getTranslationUnitDecl()->decls()) {
  if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    // Process this function
  }
}
```

### AST Matchers (Alternative Approach)

```cpp
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

// Define matchers
auto FunctionMatcher = functionDecl(
  hasAttr(attr::CUDAGlobal),  // __global__ attribute
  hasParameter(0, hasType(pointerType()))
).bind("kernel");

// Use matcher
class MatchCallback : public MatchFinder::MatchCallback {
  void run(const MatchFinder::MatchResult &Result) override {
    if (const FunctionDecl *FD = Result.Nodes.getNodeAs<FunctionDecl>("kernel")) {
      // Found a __global__ function
    }
  }
};

MatchFinder Finder;
MatchCallback Callback;
Finder.addMatcher(FunctionMatcher, &Callback);
Finder.matchAST(Context);
```

---

## Working with Function Declarations

### Finding Functions with Attributes

```cpp
bool VisitFunctionDecl(FunctionDecl *FD) {
  // Check for __global__ attribute (HIP/CUDA)
  if (FD->hasAttr<CUDAGlobalAttr>()) {
    llvm::outs() << "Found __global__ function: " << FD->getName() << "\n";
    processCUDAKernel(FD);
  }

  // Check for other attributes
  if (FD->hasAttr<CUDADeviceAttr>()) {
    llvm::outs() << "Found __device__ function\n";
  }

  return true;
}
```

### Extracting Function Information

```cpp
void processFunctionDecl(FunctionDecl *FD) {
  // Function name
  std::string name = FD->getNameAsString();

  // Return type
  QualType returnType = FD->getReturnType();

  // Number of parameters
  unsigned numParams = FD->getNumParams();

  // Iterate over parameters
  for (unsigned i = 0; i < numParams; ++i) {
    ParmVarDecl *param = FD->getParamDecl(i);

    std::string paramName = param->getNameAsString();
    QualType paramType = param->getType();

    llvm::outs() << "  Parameter " << i << ": "
                 << paramName << " : " << paramType.getAsString() << "\n";
  }

  // Function body (if available)
  if (Stmt *body = FD->getBody()) {
    // Traverse function body
  }

  // Source location
  SourceLocation loc = FD->getLocation();
  std::string filename = Context->getSourceManager()
                          .getFilename(loc).str();
  unsigned line = Context->getSourceManager()
                   .getSpellingLineNumber(loc);
}
```

---

## Extracting Type Information

### Working with QualType

```cpp
void analyzeType(QualType type, ASTContext *Context) {
  // Get canonical type (strips typedefs, etc.)
  QualType canonicalType = type.getCanonicalType();

  // Check type properties
  bool isPointer = type->isPointerType();
  bool isConst = type.isConstQualified();
  bool isVolatile = type.isVolatileQualified();
  bool isRestrict = type.isRestrictQualified();

  // Get type size and alignment
  uint64_t sizeInBits = Context->getTypeSize(type);
  uint64_t sizeInBytes = sizeInBits / 8;
  uint64_t alignInBits = Context->getTypeAlign(type);
  uint64_t alignInBytes = alignInBits / 8;

  llvm::outs() << "Type: " << type.getAsString() << "\n";
  llvm::outs() << "  Size: " << sizeInBytes << " bytes\n";
  llvm::outs() << "  Alignment: " << alignInBytes << " bytes\n";
  llvm::outs() << "  Is pointer: " << isPointer << "\n";
}
```

### Analyzing Pointer Types

```cpp
void analyzePointerType(QualType type, ASTContext *Context) {
  if (const PointerType *PT = type->getAs<PointerType>()) {
    // Get pointee type
    QualType pointeeType = PT->getPointeeType();

    llvm::outs() << "Pointer to: " << pointeeType.getAsString() << "\n";

    // Check if pointee is const
    bool isConstPointer = pointeeType.isConstQualified();

    // Get pointee size
    uint64_t pointeeSize = Context->getTypeSize(pointeeType) / 8;

    // Handle pointer-to-pointer
    if (pointeeType->isPointerType()) {
      analyzePointerType(pointeeType, Context);
    }
  }
}
```

### Analyzing Record Types (Structs/Classes)

```cpp
void analyzeRecordType(QualType type, ASTContext *Context) {
  if (const RecordType *RT = type->getAs<RecordType>()) {
    RecordDecl *RD = RT->getDecl();

    llvm::outs() << "Record type: " << RD->getName() << "\n";

    // Get record layout
    const ASTRecordLayout &layout = Context->getASTRecordLayout(RD);

    // Iterate over fields
    for (FieldDecl *field : RD->fields()) {
      std::string fieldName = field->getNameAsString();
      QualType fieldType = field->getType();

      // Get field offset
      uint64_t fieldOffsetBits = layout.getFieldOffset(field->getFieldIndex());
      uint64_t fieldOffsetBytes = fieldOffsetBits / 8;

      llvm::outs() << "  Field: " << fieldName
                   << " at offset " << fieldOffsetBytes << "\n";
    }

    // Total struct size
    uint64_t structSize = layout.getSize().getQuantity();
    llvm::outs() << "  Total size: " << structSize << " bytes\n";
  }
}
```

### Array Types

```cpp
void analyzeArrayType(QualType type, ASTContext *Context) {
  if (const ConstantArrayType *CAT = Context->getAsConstantArrayType(type)) {
    // Element type
    QualType elementType = CAT->getElementType();

    // Array size
    llvm::APInt size = CAT->getSize();
    uint64_t numElements = size.getZExtValue();

    llvm::outs() << "Array of " << numElements << " elements\n";
    llvm::outs() << "  Element type: " << elementType.getAsString() << "\n";
  }
}
```

---

## Transforming AST Nodes

### AST Rewriter

**Note:** For code transformation, use `clang::Rewriter`

```cpp
#include "clang/Rewrite/Core/Rewriter.h"

class TransformingConsumer : public ASTConsumer {
private:
  Rewriter TheRewriter;

public:
  void Initialize(ASTContext &Context) override {
    TheRewriter.setSourceMgr(Context.getSourceManager(),
                              Context.getLangOpts());
  }

  void HandleTranslationUnit(ASTContext &Context) override {
    // Perform transformations
    // ...

    // Output transformed code
    const RewriteBuffer *RewriteBuf =
      TheRewriter.getRewriteBufferFor(
        Context.getSourceManager().getMainFileID());

    if (RewriteBuf) {
      llvm::outs() << std::string(RewriteBuf->begin(), RewriteBuf->end());
    }
  }
};
```

### Rewriting Examples

```cpp
// Insert text before a location
TheRewriter.InsertTextBefore(loc, "/* INSERTED */ ");

// Insert text after a location
TheRewriter.InsertTextAfterToken(loc, " /* ADDED */");

// Replace a range
SourceRange range = stmt->getSourceRange();
TheRewriter.ReplaceText(range, "new_code");

// Remove text
TheRewriter.RemoveText(range);
```

### Practical Transformation Example

```cpp
bool VisitCallExpr(CallExpr *CE) {
  // Find calls to "threadIdx.x" and replace with Vortex equivalent
  if (MemberExpr *ME = dyn_cast<MemberExpr>(CE->getCallee())) {
    if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(ME->getBase())) {
      if (DRE->getDecl()->getName() == "threadIdx") {
        if (ME->getMemberDecl()->getName() == "x") {
          // Replace threadIdx.x with vortex_thread_id()
          SourceRange range = CE->getSourceRange();
          TheRewriter.ReplaceText(range, "vortex_thread_id()");
        }
      }
    }
  }
  return true;
}
```

---

## Generating Output

### Writing to Files

```cpp
void writeMetadataToFile(const std::string &filename,
                         const std::vector<ArgMetadata> &metadata) {
  std::error_code EC;
  llvm::raw_fd_ostream outFile(filename, EC);

  if (EC) {
    llvm::errs() << "Error opening file: " << EC.message() << "\n";
    return;
  }

  // Write header
  outFile << "// Auto-generated metadata\n";
  outFile << "#include \"vortex_hip_runtime.h\"\n\n";

  // Write metadata array
  outFile << "static const hipKernelArgumentMetadata metadata[] = {\n";
  for (const auto &arg : metadata) {
    outFile << "  {.offset = " << arg.offset
            << ", .size = " << arg.size
            << ", .alignment = " << arg.alignment
            << ", .is_pointer = " << arg.is_pointer << "},\n";
  }
  outFile << "};\n";

  outFile.close();
}
```

### Generating C++ Code

```cpp
void emitRegistrationCode(llvm::raw_ostream &OS,
                          const std::string &kernelName,
                          const std::vector<ArgMetadata> &metadata) {
  OS << "// Registration function\n";
  OS << "__attribute__((constructor))\n";
  OS << "static void register_" << kernelName << "() {\n";
  OS << "  hipError_t err = __hipRegisterFunctionWithMetadata(\n";
  OS << "    &" << kernelName << "_handle,\n";
  OS << "    \"" << kernelName << "\",\n";
  OS << "    " << kernelName << "_binary,\n";
  OS << "    " << kernelName << "_size,\n";
  OS << "    " << metadata.size() << ",\n";
  OS << "    " << kernelName << "_metadata\n";
  OS << "  );\n";
  OS << "}\n";
}
```

---

## Complete Example: HIP Metadata Extractor

### Full Plugin Implementation

```cpp
// File: HIPMetadataExtractor.cpp

#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "llvm/Support/raw_ostream.h"
#include <vector>

using namespace clang;

namespace {

// Metadata structure
struct ArgumentMetadata {
  uint64_t offset;
  uint64_t size;
  uint64_t alignment;
  bool is_pointer;
  std::string name;
  std::string type;
};

// Visitor to find __global__ functions and extract metadata
class HIPMetadataVisitor : public RecursiveASTVisitor<HIPMetadataVisitor> {
private:
  ASTContext *Context;
  std::string OutputFilename;
  llvm::raw_fd_ostream *OutFile;

public:
  explicit HIPMetadataVisitor(ASTContext *Context,
                              llvm::raw_fd_ostream *OutFile)
    : Context(Context), OutFile(OutFile) {}

  bool VisitFunctionDecl(FunctionDecl *FD) {
    // Look for __global__ functions
    if (!FD->hasAttr<CUDAGlobalAttr>()) {
      return true;
    }

    std::string kernelName = FD->getNameAsString();
    llvm::outs() << "Processing __global__ function: "
                 << kernelName << "\n";

    // Extract parameter metadata
    std::vector<ArgumentMetadata> metadata;

    // Runtime fields start at offset 0
    // User arguments start after: grid_dim(12) + block_dim(12) + shared_mem(8) = 32
    uint64_t currentOffset = 32;

    for (unsigned i = 0; i < FD->getNumParams(); ++i) {
      ParmVarDecl *param = FD->getParamDecl(i);
      QualType type = param->getType();

      ArgumentMetadata arg;
      arg.name = param->getNameAsString();
      arg.type = type.getAsString();
      arg.offset = currentOffset;
      arg.size = Context->getTypeSize(type) / 8;
      arg.alignment = Context->getTypeAlign(type) / 8;
      arg.is_pointer = type->isPointerType();

      metadata.push_back(arg);

      // Update offset for next parameter
      // Align to parameter's alignment
      currentOffset += arg.size;
      if (currentOffset % arg.alignment != 0) {
        currentOffset += arg.alignment - (currentOffset % arg.alignment);
      }

      llvm::outs() << "  Arg " << i << ": " << arg.name
                   << " (offset=" << arg.offset
                   << ", size=" << arg.size
                   << ", align=" << arg.alignment
                   << ", ptr=" << arg.is_pointer << ")\n";
    }

    // Generate metadata code
    emitMetadata(*OutFile, kernelName, metadata);

    return true;
  }

private:
  void emitMetadata(llvm::raw_ostream &OS,
                    const std::string &kernelName,
                    const std::vector<ArgumentMetadata> &metadata) {
    OS << "// Metadata for kernel: " << kernelName << "\n";
    OS << "static const hipKernelArgumentMetadata "
       << kernelName << "_metadata[] = {\n";

    for (const auto &arg : metadata) {
      OS << "  {.offset = " << arg.offset
         << ", .size = " << arg.size
         << ", .alignment = " << arg.alignment
         << ", .is_pointer = " << (arg.is_pointer ? 1 : 0)
         << "}, // " << arg.name << " : " << arg.type << "\n";
    }

    OS << "};\n\n";

    // Generate registration function
    OS << "__attribute__((constructor))\n";
    OS << "static void register_" << kernelName << "() {\n";
    OS << "  __hipRegisterFunctionWithMetadata(\n";
    OS << "    &" << kernelName << "_handle,\n";
    OS << "    \"" << kernelName << "\",\n";
    OS << "    " << kernelName << "_binary,\n";
    OS << "    " << kernelName << "_binary_size,\n";
    OS << "    " << metadata.size() << ",\n";
    OS << "    " << kernelName << "_metadata\n";
    OS << "  );\n";
    OS << "}\n\n";
  }
};

// AST Consumer
class HIPMetadataConsumer : public ASTConsumer {
private:
  HIPMetadataVisitor Visitor;
  std::unique_ptr<llvm::raw_fd_ostream> OutFile;

public:
  explicit HIPMetadataConsumer(ASTContext *Context,
                               const std::string &OutputFilename)
    : Visitor(Context, nullptr) {

    std::error_code EC;
    OutFile = std::make_unique<llvm::raw_fd_ostream>(OutputFilename, EC);

    if (EC) {
      llvm::errs() << "Error opening output file: " << EC.message() << "\n";
      return;
    }

    // Update visitor with output file
    Visitor = HIPMetadataVisitor(Context, OutFile.get());

    // Write file header
    *OutFile << "// Auto-generated HIP kernel metadata\n";
    *OutFile << "#include \"vortex_hip_runtime.h\"\n\n";
  }

  ~HIPMetadataConsumer() {
    if (OutFile) {
      OutFile->close();
    }
  }

  void HandleTranslationUnit(ASTContext &Context) override {
    Visitor.TraverseDecl(Context.getTranslationUnitDecl());
  }
};

// Plugin Action
class HIPMetadataAction : public PluginASTAction {
private:
  std::string OutputFilename = "kernel_metadata.cpp";

protected:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(
      CompilerInstance &CI, llvm::StringRef) override {
    return std::make_unique<HIPMetadataConsumer>(
      &CI.getASTContext(), OutputFilename);
  }

  bool ParseArgs(const CompilerInstance &CI,
                 const std::vector<std::string> &args) override {
    for (unsigned i = 0; i < args.size(); ++i) {
      if (args[i] == "-output" && i + 1 < args.size()) {
        OutputFilename = args[++i];
      } else {
        llvm::errs() << "Unknown argument: " << args[i] << "\n";
        return false;
      }
    }
    return true;
  }

  PluginASTAction::ActionType getActionType() override {
    return AddBeforeMainAction;
  }
};

} // namespace

// Register the plugin
static FrontendPluginRegistry::Add<HIPMetadataAction>
  X("hip-metadata", "Extract HIP kernel metadata from AST");
```

---

## Building and Using the Plugin

### CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.13.4)
project(HIPMetadataExtractor)

# Find LLVM and Clang
find_package(LLVM REQUIRED CONFIG)
find_package(Clang REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# Add definitions and include directories
add_definitions(${LLVM_DEFINITIONS})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${CLANG_INCLUDE_DIRS})

# Build plugin as shared library
add_library(HIPMetadataExtractor MODULE HIPMetadataExtractor.cpp)

# Link against Clang libraries
target_link_libraries(HIPMetadataExtractor PRIVATE
  clangAST
  clangBasic
  clangFrontend
  clangTooling
)

# Don't add 'lib' prefix
set_target_properties(HIPMetadataExtractor PROPERTIES PREFIX "")
```

### Build Instructions

```bash
# Set LLVM paths
export LLVM_DIR=/home/yaakov/vortex_hip/llvm-vortex/build/lib/cmake/llvm
export Clang_DIR=/home/yaakov/vortex_hip/llvm-vortex/build/lib/cmake/clang

# Build plugin
mkdir build && cd build
cmake -DLLVM_DIR=$LLVM_DIR -DClang_DIR=$Clang_DIR ..
make

# Result: HIPMetadataExtractor.so
```

### Using the Plugin

```bash
# Basic usage
clang++ -fplugin=./HIPMetadataExtractor.so -c kernel.cpp

# With plugin arguments
clang++ -fplugin=./HIPMetadataExtractor.so \
  -plugin-arg-hip-metadata -output \
  -plugin-arg-hip-metadata custom_metadata.cpp \
  -c kernel.hip

# With RISC-V target (for Vortex)
$LLVM_VORTEX/bin/clang++ \
  -fplugin=./HIPMetadataExtractor.so \
  -target riscv32 -march=rv32imaf \
  --sysroot=$RISCV_SYSROOT \
  -c kernel.hip
```

---

## Debugging and Testing

### Enable Plugin Debugging

```cpp
// In your plugin
#define DEBUG_TYPE "hip-metadata"

#include "llvm/Support/Debug.h"

// Use LLVM_DEBUG macro
LLVM_DEBUG(llvm::dbgs() << "Processing function: " << FD->getName() << "\n");
```

### Run with Debug Output

```bash
# Enable LLVM debug output
clang++ -fplugin=./plugin.so -mllvm -debug-only=hip-metadata -c kernel.cpp

# Verbose plugin loading
clang++ -fplugin=./plugin.so -v -c kernel.cpp
```

### Print AST for Inspection

```cpp
// Dump entire AST
Context.getTranslationUnitDecl()->dump();

// Dump specific node
FD->dump();

// Pretty-print with source locations
FD->dump(llvm::outs(), Context.getSourceManager());
```

### Test Cases

```cpp
// Test input: test_kernel.hip
__global__ void simple_kernel(float *a, float *b, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    a[idx] = b[idx] * 2.0f;
  }
}
```

```bash
# Run plugin
clang++ -fplugin=./HIPMetadataExtractor.so -c test_kernel.hip

# Check output
cat kernel_metadata.cpp
```

### Expected Output

```cpp
// Metadata for kernel: simple_kernel
static const hipKernelArgumentMetadata simple_kernel_metadata[] = {
  {.offset = 32, .size = 4, .alignment = 4, .is_pointer = 1}, // a : float *
  {.offset = 36, .size = 4, .alignment = 4, .is_pointer = 1}, // b : float *
  {.offset = 40, .size = 4, .alignment = 4, .is_pointer = 0}, // n : int
};
```

---

## Summary

### Key Clang Plugin APIs

| Component | API | Purpose |
|-----------|-----|---------|
| **Entry Point** | `PluginASTAction` | Register plugin with Clang |
| **AST Access** | `ASTConsumer` | Process translation unit |
| **Traversal** | `RecursiveASTVisitor` | Walk AST nodes |
| **Functions** | `FunctionDecl` | Access function information |
| **Parameters** | `ParmVarDecl` | Access parameter information |
| **Types** | `QualType`, `ASTContext` | Query type properties |
| **Attributes** | `hasAttr<AttrType>()` | Check for attributes |
| **Transformation** | `Rewriter` | Modify source code |
| **Output** | `raw_fd_ostream` | Write generated files |

### Phase 2 Plugin Workflow

```
HIP Source (kernel.hip)
    ↓
[Clang Frontend + Plugin]
    ├─> Parse to AST
    ├─> Run Semantic Analysis
    ├─> ⭐ Plugin Executes:
    │   1. Find __global__ functions
    │   2. Extract parameter metadata (CORRECT offsets!)
    │   3. Generate kernel_metadata.cpp
    │   4. Transform HIP → Vortex (optional)
    ↓
Continue to CodeGen → RISC-V → Vortex Binary
```

### Resources

**Official Clang Documentation:**
- Plugin Tutorial: https://clang.llvm.org/docs/ClangPlugins.html
- AST Intro: https://clang.llvm.org/docs/IntroductionToTheClangAST.html
- AST Matchers: https://clang.llvm.org/docs/LibASTMatchers.html

**LLVM API Reference:**
- Doxygen: https://clang.llvm.org/doxygen/

**Example Plugins:**
- `llvm-project/clang/examples/`
- `llvm-project/clang-tools-extra/`

---

**Document Status:** Complete Guide
**Last Updated:** 2025-11-09
**Next Steps:** Implement HIP metadata extractor plugin for Phase 2
