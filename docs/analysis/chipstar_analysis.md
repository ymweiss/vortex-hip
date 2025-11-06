# chipStar Runtime Analysis: SPIR-V Loading, Kernel Management, and Compiler Interface

## Table of Contents
1. [SPIR-V Binary Loading and Execution](#spirv-binary-loading-and-execution)
2. [Kernel Registration and Management](#kernel-registration-and-management)
3. [Kernel Argument Passing](#kernel-argument-passing)
4. [Compiler-Runtime Interface](#compiler-runtime-interface)

---

## SPIR-V Binary Loading and Execution

### Overview

chipStar loads SPIR-V binaries through a multi-stage pipeline involving extraction, registration, lazy finalization, and backend compilation. The runtime uses a sophisticated lazy evaluation strategy to defer expensive compilation until kernels are actually needed.

### Key Components

#### 1. Binary Extraction (`spirv-extractor`)

**Location:** `/home/yaakov/vortex_hip/chipStar/tools/spirv-extractor/spirv-extractor.hh`

The extractor handles two formats:
- **Clang offload bundle** format with bundle ID `hip-spirv64`
- **Direct SPIR-V** binaries (magic number `0x07230203`)

```cpp
// Extracts SPIR-V from either ELF-wrapped or direct SPIR-V binaries
std::string_view extractSPIRVModule(const void *FatBinary, std::string &ErrorMsg);
```

#### 2. SPIR-V Registration (`SPVRegister`)

**Location:** `/home/yaakov/vortex_hip/chipStar/src/SPVRegister.cc`

The global `SPVRegister` singleton manages all SPIR-V modules before backend compilation.

**Key Registration Function:**
```cpp
// From CHIPBindings.cc:6085-6138
extern "C" void **__hipRegisterFatBinary(const void *Data) {
  // 1. Extract SPIR-V module from fat binary wrapper
  auto SPIRVModuleSpan = extractSPIRVModule(Wrapper->binary, ErrorMsg);

  // 2. Register with global SPVRegister
  SPVRegister::Handle ModHandle =
      getSPVRegister().registerSource(SPIRVModuleSpan);

  // 3. Optionally compile immediately (if CHIP_LAZY_JIT=0)
  if (!ChipEnvVars.getLazyJit()) {
    const SPVModule *SMod = getSPVRegister().getSource(ModHandle);
    Backend->getActiveDevice()->getOrCreateModule(*SMod);
  }

  return (void **)ModHandle.Module;
}
```

**SPVRegister Class Structure:**
```cpp
class SPVRegister {
private:
  std::mutex Mtx_;
  std::set<std::unique_ptr<SPVModule>> Sources_;
  std::unordered_map<const void *, SPVGlobalObject *> HostPtrLookup_;

public:
  Handle registerSource(std::string_view SourceModule);
  void bindFunction(Handle, HostPtr, const char *Name);
  const SPVModule *getSource(HostPtr);
};
```

From `SPVRegister.cc:47-54`:
```cpp
SPVRegister::Handle SPVRegister::registerSource(std::string_view SourceModule) {
  LOCK(Mtx_);
  auto Ins = Sources_.emplace(std::make_unique<SPVModule>());
  SPVModule *SrcMod = Ins.first->get();
  SrcMod->OriginalBinary_ = SourceModule;  // Store original SPIR-V
  return Handle{reinterpret_cast<void *>(SrcMod)};
}
```

#### 3. Lazy SPIR-V Finalization

**Location:** `/home/yaakov/vortex_hip/chipStar/src/SPVRegister.cc:159-211`

SPIR-V finalization is deferred until first access. The finalization pipeline has three stages:

```cpp
SPVModule *SPVRegister::getFinalizedSource(SPVModule *SrcMod) {
  if (SrcMod->FinalizedBinary_.size())  // Already cached
    return SrcMod;

  std::vector<uint32_t> Binary;

  // Stage 1: Preprocessing - Clean and normalize SPIR-V
  if (!preprocessSPIRV(SrcMod->OriginalBinary_.data(),
                       SrcMod->OriginalBinary_.size(),
                       PreventNameDemangling, Binary)) {
    CHIPERR_LOG_AND_THROW("SPIR-V preprocessing failure.", hipErrorTbd);
  }

  // Stage 2: Analysis - Extract kernel metadata
  if (!analyzeSPIRV(Binary.data(), Binary.size(), SrcMod->ModuleInfo_)) {
    CHIPERR_LOG_AND_THROW("SPIR-V analysis failure.", hipErrorTbd);
  }

  // Stage 3: Postprocessing - Remove chipStar runtime metadata
  if (!postprocessSPIRV(Binary)) {
    CHIPERR_LOG_AND_THROW("SPIR-V postprocessing failure.", hipErrorTbd);
  }

  SrcMod->FinalizedBinary_ = std::move(Binary);
  return SrcMod;
}
```

**Preprocessing Stage:**
- Header validation
- Extension filtering
- Name demangling for PowerVR GPUs (swap first two chars of mangled names)
- Platform-specific workarounds (Intel Compute Runtime, Mali GPU, llvm-spirv)

**Analysis Stage** (`spirv.cc:1017-1022`):
- Parses SPIR-V into `SPIRVmodule` class representation
- Extracts entry points (kernels)
- Builds type system (`SPIRVtypePOD`, `SPIRVtypePointer`, etc.)
- Infers argument kinds and sizes
- Calls `fillModuleInfo()` to create `SPVModuleInfo`

**Postprocessing Stage:**
- Removes chipStar-specific metadata annotations
- Prepares final SPIR-V for backend consumption

#### 4. Backend Compilation

**OpenCL Backend** (`CHIPBackendOpenCL.cc:1169-1264`):

```cpp
void CHIPModuleOpenCL::compile(chipstar::Device *ChipDev) {
  // Get finalized SPIR-V binary
  auto SrcBin = Src_->getBinary();

  // Compile SPIR-V to OpenCL program
  cl::Program ClMainObj = compileIL(*ChipCtxOcl->get(), *ChipDevOcl,
                                    SrcBin.data(), SrcBin.size(), ...);

  // Link against device library (builtins)
  Program_ = cl::linkProgram(ClObjects, ...);

  // Create OpenCL kernels from program
  std::vector<cl::Kernel> Kernels;
  Program_.createKernels(&Kernels);

  // Wrap each OpenCL kernel
  for (auto &Krnl : Kernels) {
    std::string HostFName;
    Krnl.getInfo(CL_KERNEL_FUNCTION_NAME, &HostFName);

    // Lookup metadata
    auto *FuncInfo = findFunctionInfo(HostFName);
    if (!FuncInfo) continue;  // Not a user kernel

    // Create chipStar kernel wrapper
    CHIPKernelOpenCL *ChipKernel =
        new CHIPKernelOpenCL(Krnl, ChipDevOcl, HostFName, FuncInfo, this);

    addKernel(ChipKernel);
  }
}
```

**Level Zero Backend:**
- Uses `zeModuleCreate()` to compile SPIR-V
- Similar kernel wrapping pattern

### SPIR-V Loading Flow Diagram

```
hipModuleLoadData() / __hipRegisterFatBinary()
    |
    v
extractSPIRVModule()  [Extract from ELF or direct SPIR-V]
    |
    v
SPVRegister::registerSource()  [Store in SPVModule::OriginalBinary_]
    |
    v
[LAZY - Deferred until first access]
    |
    v
SPVRegister::getFinalizedSource()
    |
    +---> preprocessSPIRV()  [Clean, normalize]
    |
    +---> analyzeSPIRV()     [Extract metadata]
    |
    +---> postprocessSPIRV() [Remove annotations]
    |
    v
Module::compile()  [Backend-specific compilation]
    |
    +---> OpenCL: clCreateProgramWithIL() → clBuildProgram()
    |
    +---> Level0: zeModuleCreate()
    |
    v
Create Kernel wrappers with metadata
    |
    v
Ready for hipLaunchKernel()
```

### Key Design Features

- **Lazy Compilation:** Controlled by `CHIP_LAZY_JIT` environment variable
- **Metadata Extraction:** Rich type information preserved throughout pipeline
- **Backend Abstraction:** Same SPIR-V serves OpenCL and Level Zero
- **Thread Safety:** `SPVRegister` uses mutex for concurrent access
- **Platform Workarounds:** Handles vendor-specific quirks

---

## Kernel Registration and Management

### Overview

chipStar maintains a two-tier kernel management system: a global `SPVRegister` for source-level registration, and per-device `Module` objects for compiled kernels.

### Registration Flow

#### Phase 1: Function Binding

**Location:** `/home/yaakov/vortex_hip/chipStar/src/CHIPBindings.cc:6162-6182`

```cpp
extern "C" void __hipRegisterFunction(void **Data, const void *HostFunction,
                                      char *DeviceFunction,
                                      const char *FuncDeviceName, ...) {
  SPVRegister::Handle ModHandle{reinterpret_cast<void *>(Data)};
  getSPVRegister().bindFunction(ModHandle, HostPtr(HostFunction),
                                FuncDeviceName);
}
```

From `SPVRegister.cc:58-91`:
```cpp
void SPVRegister::bindFunction(Handle Handle, HostPtr Ptr, const char *Name) {
  LOCK(Mtx_);
  auto *SrcMod = reinterpret_cast<SPVModule *>(Handle.Module);

  std::string FuncName(Name);
  if (PreventNameDemangling) {
    std::swap(FuncName[0], FuncName[1]);  // PowerVR workaround
  }

  if (HostPtrLookup_.count(Ptr)) {
    // Duplicate registration from template/inline __global__ functions
    // across multiple translation units. Ignore duplicates.
    return;
  }

  // Create function entry in module
  SrcMod->Kernels.emplace_back(SPVFunction{SrcMod, Ptr, std::move(FuncName)});

  // Add to lookup map
  HostPtrLookup_.emplace(std::make_pair(Ptr, &SrcMod->Kernels.back()));
}
```

#### Phase 2: Metadata Extraction

**Location:** `/home/yaakov/vortex_hip/chipStar/src/spirv.cc:590-616`

During SPIR-V analysis, `fillModuleInfo()` extracts kernel signatures:

```cpp
void SPIRVmodule::fillModuleInfo(SPVModuleInfo &ModuleInfo) {
  for (auto i : EntryPoints_) {  // For each kernel
    InstWord EntryPointID = i.first;
    std::string_view KernelName = i.second;

    // Get function metadata from SPIR-V analysis
    auto FnInfo = KernelInfoMap_[EntryPointID];

    // Check for PODByRef (spilled) arguments
    if (SpilledArgAnnotations_.count(KernelName)) {
      for (auto &Kv : SpilledArgAnnotations_[KernelName]) {
        FnInfo->ArgTypeInfo_[Kv.first].Kind = SPVTypeKind::PODByRef;
        FnInfo->ArgTypeInfo_[Kv.first].Size = Kv.second;
      }
    }

    // Add to module's function info map
    ModuleInfo.FuncInfoMap.emplace(std::make_pair(KernelName, FnInfo));
  }
}
```

### Kernel Metadata Structures

#### SPVFuncInfo (Argument Metadata)

**Location:** `/home/yaakov/vortex_hip/chipStar/src/SPIRVFuncInfo.hh:72-124`

```cpp
class SPVFuncInfo {
  std::vector<SPVArgTypeInfo> ArgTypeInfo_;
  bool HasByRefArgs_ = false;

public:
  // Visitor methods for argument iteration
  void visitClientArgs(ClientArgVisitor Fn);  // User-visible args only
  void visitKernelArgs(KernelArgVisitor Fn);  // All args including compiler-generated

  unsigned getNumClientArgs() const;
  unsigned getNumKernelArgs() const { return ArgTypeInfo_.size(); }
  bool hasByRefArgs() const noexcept { return HasByRefArgs_; }
};
```

#### SPVArgTypeInfo (Per-Argument Metadata)

From `SPIRVFuncInfo.hh:33-70`:

```cpp
enum class SPVTypeKind : unsigned {
  POD,      // Primitive or aggregate by-value
  Pointer,  // Any pointer type
  PODByRef, // Large structure passed via device buffer
  Image,    // Texture image handle
  Sampler,  // Texture sampler handle
  Opaque,   // Unresolved special type
  Unknown
};

enum class SPVStorageClass : unsigned {
  Private = 0,
  CrossWorkgroup = 1,
  UniformConstant = 2,
  Workgroup = 3,
  Unknown = 1000
};

struct SPVArgTypeInfo {
  SPVTypeKind Kind;
  SPVStorageClass StorageClass;
  size_t Size;

  bool isWorkgroupPtr() const {
    return Kind == SPVTypeKind::Pointer &&
           StorageClass == SPVStorageClass::Workgroup;
  }
};
```

### Kernel Management Data Structures

#### SPVModule (Source Module)

**Location:** `/home/yaakov/vortex_hip/chipStar/src/SPVRegister.hh:70-100`

```cpp
class SPVModule {
public:
  std::list<SPVFunction> Kernels;     // Registered kernels
  std::list<SPVVariable> Variables;   // Device global variables
  bool HasAbortFlag = false;

private:
  std::string_view OriginalBinary_;        // Raw SPIR-V from compiler
  std::vector<uint32_t> FinalizedBinary_;  // After preprocessing/analysis
  SPVModuleInfo ModuleInfo_;               // Extracted metadata

  std::string_view getBinary() const;
  const SPVModuleInfo &getInfo() const;
};
```

#### Module (Compiled Module)

**Location:** `/home/yaakov/vortex_hip/chipStar/src/CHIPBackend.hh:901-1046`

```cpp
class Module {
private:
  std::vector<chipstar::Kernel *> ChipKernels_;    // Compiled kernels
  std::vector<chipstar::DeviceVar *> ChipVars_;    // Device variables
  const SPVModule *Src_;                           // Source SPIR-V module
  std::once_flag Compiled_;                        // Lazy compilation flag

public:
  void addKernel(chipstar::Kernel *Kernel);
  void compileOnce(chipstar::Device *ChipDev);
  virtual void compile(chipstar::Device *ChipDev) = 0;  // Backend-specific

  chipstar::Kernel *findKernel(const std::string &Name);
  chipstar::Kernel *getKernel(const void *HostFPtr);
  SPVFuncInfo *findFunctionInfo(const std::string &FName);
};
```

#### Kernel (Compiled Kernel)

**Location:** `/home/yaakov/vortex_hip/chipStar/src/CHIPBackend.hh:1051-1129`

```cpp
class Kernel {
protected:
  std::string HostFName_;           // Kernel function name
  const void *HostFPtr_ = nullptr;  // Host function pointer
  const void *DevFPtr_;             // Device function pointer
  SPVFuncInfo *FuncInfo_;           // Argument metadata

public:
  std::string getName();
  const void *getHostPtr();
  SPVFuncInfo *getFuncInfo();

  void setHostPtr(const void *HostFPtr);

  virtual hipError_t getAttributes(hipFuncAttributes *Attr) = 0;
};
```

### Kernel Lookup Mechanisms

#### 1. By Host Function Pointer

From `CHIPBackend.cc:315-329`:

```cpp
chipstar::Kernel *chipstar::Module::getKernel(const void *HostFPtr) {
  for (auto &Kernel : ChipKernels_) {
    if (Kernel->getHostPtr() == HostFPtr)
      return Kernel;
  }
  CHIPERR_LOG_AND_THROW("Kernel not found", hipErrorLaunchFailure);
}
```

#### 2. By Kernel Name

From `CHIPBackend.cc:296-311`:

```cpp
chipstar::Kernel *chipstar::Module::findKernel(const std::string &Name) {
  auto KernelFound = std::find_if(ChipKernels_.begin(), ChipKernels_.end(),
    [&Name](chipstar::Kernel *Kernel) {
      return Kernel->getName().compare(Name) == 0;
    });
  return KernelFound == ChipKernels_.end() ? nullptr : *KernelFound;
}
```

#### 3. Via Global Register

From `CHIPBackend.cc:1015-1038`:

```cpp
chipstar::Module *chipstar::Device::getOrCreateModule(HostPtr Ptr) {
  // Check device-level cache
  if (HostPtrToCompiledMod_.count(Ptr))
    return HostPtrToCompiledMod_[Ptr];

  // Get source module from global register
  auto *SrcMod = getSPVRegister().getSource(Ptr);
  if (!SrcMod) return nullptr;

  // Compile source module
  auto *Mod = getOrCreateModule(*SrcMod);

  // Bind host pointers to compiled kernels
  for (const auto &Info : SrcMod->Kernels) {
    std::string NameTmp(Info.Name.begin(), Info.Name.end());
    chipstar::Kernel *Kernel = Mod->getKernelByName(NameTmp);
    Kernel->setHostPtr(Info.Ptr);
    HostPtrToCompiledMod_[Info.Ptr] = Mod;
  }

  return Mod;
}
```

### Metadata Hierarchy

```
SPVRegister (Global Singleton)
├── SPVModule (per binary)
│   ├── OriginalBinary_ (pre-finalization)
│   ├── FinalizedBinary_ (after preprocessing/analysis)
│   ├── SPVModuleInfo
│   │   └── FuncInfoMap: "kernel_name" -> SPVFuncInfo
│   │       ├── ArgTypeInfo_[0..N]
│   │       │   ├── Kind (POD, Pointer, Image, etc.)
│   │       │   ├── StorageClass
│   │       │   └── Size
│   │       └── HasByRefArgs_
│   ├── Kernels (list<SPVFunction>)
│   │   └── SPVFunction{Parent, Ptr, Name}
│   └── Variables (list<SPVVariable>)
│
├── HostPtrLookup_: HostPtr -> SPVGlobalObject*
│
Device (per GPU)
├── SrcModToCompiledMod_: SPVModule* -> Module*
├── HostPtrToCompiledMod_: HostPtr -> Module*
└── Module (compiled)
    ├── ChipKernels_: vector<Kernel*>
    │   └── Kernel{HostFName_, HostFPtr_, DevFPtr_, FuncInfo_}
    └── ChipVars_: vector<DeviceVar*>
```

### Complete Registration Flow

```
1. __hipRegisterFatBinary()
   ↓ Extract SPIR-V from fat binary

2. SPVRegister::registerSource()
   ↓ Store in SPVModule::OriginalBinary_

3. __hipRegisterFunction()
   ↓ Map host function ptr -> kernel name

4. [LAZY] First kernel access
   ↓ Trigger finalization

5. analyzeSPIRV()
   ↓ Parse kernel entry points and argument types

6. fillModuleInfo()
   ↓ Create SPVModuleInfo with function metadata

7. Module::compile()
   ↓ Backend compiles SPIR-V

8. Create Kernel wrappers
   ↓ Attach FuncInfo pointers

9. Kernel::setHostPtr()
   ↓ Bind host function pointers

10. Ready for lookup and launch
```

---

## Kernel Argument Passing

### Overview

chipStar uses a visitor pattern to iterate through kernel arguments and marshal them to backend-specific APIs. The system handles multiple argument types with special handling for large structures, textures, and dynamic shared memory.

### Argument Setting Interface

#### SPVFuncInfo Visitor Pattern

**Location:** `/home/yaakov/vortex_hip/chipStar/src/SPIRVFuncInfo.hh:82-124`

```cpp
class SPVFuncInfo {
  struct Arg : SPVArgTypeInfo {
    size_t Index;       // Argument index
    const void *Data;   // Address to argument value
  };

  using ClientArgVisitor = std::function<void(const ClientArg &)>;
  using KernelArgVisitor = std::function<void(const KernelArg &)>;

  // Iterate user-visible arguments only
  void visitClientArgs(void **ArgList, ClientArgVisitor Fn) const;

  // Iterate all arguments including compiler-generated
  void visitKernelArgs(void **ArgList, KernelArgVisitor Fn) const;
};
```

**Two Visitor Types:**
- **ClientArgs:** Skips compiler-generated args (Image, Sampler, Workgroup pointers)
- **KernelArgs:** Visits all arguments for actual kernel launch

#### ExecItem (Execution Configuration)

**Location:** `/home/yaakov/vortex_hip/chipStar/src/CHIPBackend.hh`

```cpp
class ExecItem {
protected:
  void **Args_;                 // Pointer array to argument values
  dim3 GridDim_, BlockDim_;
  size_t SharedMem_;
  chipstar::Queue *ChipQueue_;
  std::shared_ptr<chipstar::ArgSpillBuffer> ArgSpillBuffer_;

public:
  virtual void setupAllArgs() = 0;  // Backend-specific argument setup
};
```

### Backend-Specific Argument Setup

#### OpenCL Backend

**Location:** `/home/yaakov/vortex_hip/chipStar/src/backend/OpenCL/CHIPBackendOpenCL.cc:2036-2147`

```cpp
void CHIPExecItemOpenCL::setupAllArgs() {
  CHIPKernelOpenCL *Kernel = (CHIPKernelOpenCL *)getKernel();
  SPVFuncInfo *FuncInfo = Kernel->getFuncInfo();

  // Handle PODByRef arguments (large structures)
  if (FuncInfo->hasByRefArgs()) {
    ArgSpillBuffer_ =
        std::make_shared<chipstar::ArgSpillBuffer>(ChipQueue_->getContext());
    ArgSpillBuffer_->computeAndReserveSpace(*FuncInfo);
  }

  cl_kernel KernelHandle = ClKernel_.get()->get();

  // Visit each kernel argument
  auto ArgVisitor = [&](const SPVFuncInfo::KernelArg &Arg) -> void {
    switch (Arg.Kind) {
    case SPVTypeKind::Image: {
      auto *TexObj = *reinterpret_cast<const CHIPTextureOpenCL *const *>(Arg.Data);
      cl_mem Image = TexObj->getImage();
      clSetKernelArg(KernelHandle, Arg.Index, sizeof(cl_mem), &Image);
      break;
    }

    case SPVTypeKind::Sampler: {
      auto *TexObj = *reinterpret_cast<const CHIPTextureOpenCL *const *>(Arg.Data);
      cl_sampler Sampler = TexObj->getSampler();
      clSetKernelArg(KernelHandle, Arg.Index, sizeof(cl_sampler), &Sampler);
      break;
    }

    case SPVTypeKind::POD: {
      // Direct value copy
      clSetKernelArg(KernelHandle, Arg.Index, Arg.Size, Arg.Data);
      break;
    }

    case SPVTypeKind::Pointer: {
      if (Arg.isWorkgroupPtr()) {
        // Dynamic shared memory: pass size with nullptr
        clSetKernelArg(KernelHandle, Arg.Index, SharedMem_, nullptr);
      } else {
        auto *DevPtr = *reinterpret_cast<const void *const *>(Arg.Data);

        // Two allocation strategies:
        if (Ctx->getAllocStrategy() == AllocationStrategy::BufferDevAddr) {
          // Buffer device addresses
          Ctx->clSetKernelArgDevicePointerEXT(KernelHandle, Arg.Index, DevPtr);
        } else {
          // SVM pointers
          clSetKernelArgSVMPointer(KernelHandle, Arg.Index, DevPtr);
        }
      }
      break;
    }

    case SPVTypeKind::PODByRef: {
      // Large struct: pass pointer to spill buffer
      auto *SpillSlot = ArgSpillBuffer_->allocate(Arg);
      clSetKernelArgSVMPointer(KernelHandle, Arg.Index, SpillSlot);
      break;
    }
    }
  };

  FuncInfo->visitKernelArgs(getArgs(), ArgVisitor);

  // Copy spilled arguments to device
  if (FuncInfo->hasByRefArgs())
    ChipQueue_->memCopyAsync(ArgSpillBuffer_->getDeviceBuffer(),
                             ArgSpillBuffer_->getHostBuffer(),
                             ArgSpillBuffer_->getSize(),
                             hipMemcpyHostToDevice);
}
```

#### Level Zero Backend

**Location:** `/home/yaakov/vortex_hip/chipStar/src/backend/Level0/CHIPBackendLevel0.cc:2964-3058`

Similar structure but uses `zeKernelSetArgumentValue()` instead of OpenCL APIs:

```cpp
void CHIPExecItemLevel0::setupAllArgs() {
  // Similar spill buffer setup...

  auto ArgVisitor = [&](const SPVFuncInfo::KernelArg &Arg) -> void {
    switch (Arg.Kind) {
    case SPVTypeKind::Image:
      zeKernelSetArgumentValue(Kernel->get(), Arg.Index,
                              sizeof(ze_image_handle_t), &ImageHandle);
      break;

    case SPVTypeKind::Sampler:
      zeKernelSetArgumentValue(Kernel->get(), Arg.Index,
                              sizeof(ze_sampler_handle_t), &SamplerHandle);
      break;

    case SPVTypeKind::POD:
    case SPVTypeKind::Pointer:
      if (Arg.isWorkgroupPtr()) {
        zeKernelSetArgumentValue(Kernel->get(), Arg.Index, SharedMem_, nullptr);
      } else {
        zeKernelSetArgumentValue(Kernel->get(), Arg.Index, ArgSize, ArgData);
      }
      break;

    case SPVTypeKind::PODByRef:
      auto *SpillSlot = ArgSpillBuffer_->allocate(Arg);
      zeKernelSetArgumentValue(Kernel->get(), Arg.Index, sizeof(void *), &SpillSlot);
      break;
    }
  };

  FuncInfo->visitKernelArgs(getArgs(), ArgVisitor);

  // Async copy spill buffer
  if (FuncInfo->hasByRefArgs())
    ChipQueue_->memCopyAsync(...);
}
```

### Argument Buffer Management (ArgSpillBuffer)

**Purpose:** Handles large structures (PODByRef) that exceed kernel parameter register limits.

**Location:** `/home/yaakov/vortex_hip/chipStar/src/CHIPBackend.cc:495-529`

```cpp
class ArgSpillBuffer {
  chipstar::Context *Ctx_;
  std::unique_ptr<char[]> HostBuffer_;
  char *DeviceBuffer_;
  std::map<size_t, size_t> ArgIndexToOffset_;  // Arg index -> buffer offset
  size_t Size_;

public:
  void computeAndReserveSpace(const SPVFuncInfo &KernelInfo) {
    size_t Offset = 0;

    // Iterate through PODByRef arguments
    for (each PODByRef arg) {
      Offset = roundUp(Offset, 32);  // Align to sizeof(double4)
      ArgIndexToOffset_[Arg.Index] = Offset;
      Offset += Arg.Size;
    }

    // Single host allocation
    HostBuffer_ = std::make_unique<char[]>(Offset);

    // Single device allocation
    DeviceBuffer_ = Ctx_->allocate(Offset, 32, hipMemoryTypeDevice);
  }

  void *allocate(const SPVFuncInfo::Arg &Arg) {
    // Copy value to host buffer
    auto *HostPtr = HostBuffer_.get() + ArgIndexToOffset_[Arg.Index];
    std::memcpy(HostPtr, Arg.Data, Arg.Size);

    // Return device buffer pointer
    return DeviceBuffer_ + ArgIndexToOffset_[Arg.Index];
  }
};
```

**Allocation Strategy:**
1. **computeAndReserveSpace():** Calculate total size, allocate single buffers
2. **allocate():** Copy individual arguments to host buffer, return device offsets
3. **Async Copy:** Transfer entire buffer host→device after all arguments set

### Special Argument Handling

#### Scalar Arguments (POD)
- Passed by value directly
- Both size and value provided to backend

#### Pointer Arguments
- **Device pointers:** Backend-dependent mechanism
  - OpenCL SVM: `clSetKernelArgSVMPointer()`
  - OpenCL Buffer: `clSetKernelArgDevicePointerEXT()`
  - Level0: `zeKernelSetArgumentValue()`
- **Workgroup pointers (dynamic shared memory):**
  - OpenCL: `clSetKernelArg(kernel, index, size, nullptr)`
  - Level0: `zeKernelSetArgumentValue(kernel, index, size, nullptr)`
- **Null pointers:** Gracefully handled with fallback

#### Texture Objects
- Compiler converts `hipTextureObject_t` → Image + Sampler pair
- Two kernel arguments per texture:
  - Image argument (`SPVTypeKind::Image`)
  - Sampler argument (`SPVTypeKind::Sampler`)

#### Large Structures (PODByRef)
- Compiler pass (`HipKernelArgSpiller`) marks large args for indirect passing
- Runtime allocates spill buffer, copies structure, passes pointer

### Kernel Launch Flow

```
hipLaunchKernel(kernel, grid, block, args, shared)
    ↓
Queue::launchKernel()
    ↓
Backend::createExecItem(grid, block, shared, queue)
    ↓
ExecItem::setKernel(kernel)
    ↓
ExecItem::setArgs(args)  // Store arg array pointer
    ↓
ExecItem::setupAllArgs()  // Backend-specific marshalling
    ↓
    FuncInfo->visitKernelArgs() iterates arguments:
        POD → clSetKernelArg(size, value)
        Pointer → clSetKernelArgSVMPointer(ptr)
        Workgroup → clSetKernelArg(size, nullptr)
        Image/Sampler → clSetKernelArg(handle)
        PODByRef → Allocate spill buffer, copy, pass pointer
    ↓
    If PODByRef args exist:
        Async copy spill buffer host→device
    ↓
Queue::launch(ExecItem)  // Actual backend kernel execution
    ↓
Cleanup ExecItem
```

---

## Compiler-Runtime Interface

### Overview

The chipStar compiler-runtime interface is based on SPIR-V as the intermediate representation, with rich metadata annotations that communicate kernel signatures, argument types, and execution requirements.

### Compiler Output Format

#### SPIR-V Binary Structure

**Format:** Standard SPIR-V 1.0+ binary
- **Magic Number:** `0x07230203`
- **Execution Model:** `Kernel`
- **Memory Model:** `OpenCL`
- **Addressing Model:** `Physical64`

**Packaging:** Clang offload bundle format
- **Bundle ID:** `hip-spirv64`
- **ELF Section:** `.hip_fatbin`
- **Wrapper:** `__CudaFatBinaryWrapper` structure

**Extraction Code** (`spirv-extractor.hh:134-218`):
```cpp
std::string_view extractSPIRVModule(const void *FatBinary, std::string &ErrorMsg) {
  // Try direct SPIR-V (magic number check)
  if (isSPIRV(FatBinary))
    return std::string_view(reinterpret_cast<const char *>(FatBinary), Size);

  // Try Clang offload bundle
  auto Bundle = ClangOffloadBundleUnpacker(FatBinary, Size);
  if (auto SPIRVEntry = Bundle.getEntry("hip-spirv64"))
    return *SPIRVEntry;

  ErrorMsg = "Not a valid SPIR-V module or offload bundle";
  return {};
}
```

### Compiler Metadata Annotations

#### Entry Point Declarations

SPIR-V `OpEntryPoint` instructions declare kernels:
```
OpEntryPoint Kernel %kernel_func "kernel_name" ...
OpExecutionMode %kernel_func LocalSize 256 1 1
```

#### Argument Type Information

Embedded in SPIR-V type system:
- `OpTypeInt`, `OpTypeFloat` → `SPVTypeKind::POD`
- `OpTypePointer` → `SPVTypeKind::Pointer` with `StorageClass`
- `OpTypeImage` → `SPVTypeKind::Image`
- `OpTypeSampler` → `SPVTypeKind::Sampler`

**Storage Classes:**
- `CrossWorkgroup` (Global) → Device memory pointers
- `Workgroup` → Shared memory pointers
- `UniformConstant` → Textures/samplers
- `Private` → Thread-local

#### PODByRef (Spilled Argument) Annotations

**Compiler Pass:** `HipKernelArgSpiller.cpp`

Adds metadata to SPIR-V indicating which arguments exceed size thresholds:
```
OpDecorate %arg ByRefArg <size>
```

**Runtime Extraction** (`spirv.cc:590-616`):
```cpp
if (SpilledArgAnnotations_.count(KernelName)) {
  for (auto &Kv : SpilledArgAnnotations_[KernelName]) {
    FnInfo->ArgTypeInfo_[Kv.first].Kind = SPVTypeKind::PODByRef;
    FnInfo->ArgTypeInfo_[Kv.first].Size = Kv.second;
  }
}
```

### LLVM Compiler Passes

**Location:** `/home/yaakov/vortex_hip/chipStar/llvm_passes/`

Key passes that annotate SPIR-V:

1. **HipDynMem.cpp** - Shared memory metadata
   - Converts dynamic shared memory pointer to workgroup storage class
   - Adds kernel argument for shared memory size

2. **HipTextureLowering.cpp** - Texture handling
   - Converts `hipTextureObject_t` → Image + Sampler arguments
   - Lowers texture intrinsics to SPIR-V image ops

3. **HipKernelArgSpiller.cpp** - Large argument handling
   - Identifies POD arguments > threshold (e.g., 128 bytes)
   - Marks for indirect passing via device buffer
   - Adds `ByRefArg` annotations

4. **HipGlobalVariables.cpp** - Device variables
   - Handles `__device__` and `__constant__` variables
   - Creates shadow host objects for binding

5. **HipIGBADetector.cpp** - Memory access analysis
   - Detects indirect global buffer accesses
   - Sets `HasNoIGBAs` flag in `SPVModuleInfo`

6. **HipAbort.cpp** - Abort flag support
   - Adds global abort flag for device-side `assert()`
   - Integrates with runtime abort checking

### API Boundary: Registration Functions

**Location:** `/home/yaakov/vortex_hip/chipStar/src/CHIPBindings.cc:6085-6247`

Three primary registration functions called by compiler-generated code:

#### 1. __hipRegisterFatBinary

```cpp
extern "C" void **__hipRegisterFatBinary(const void *Data) {
  const __CudaFatBinaryWrapper *Wrapper =
      reinterpret_cast<const __CudaFatBinaryWrapper *>(Data);

  // Extract SPIR-V from bundle
  auto SPIRVModuleSpan = extractSPIRVModule(Wrapper->binary, ErrorMsg);

  // Register with global SPVRegister
  SPVRegister::Handle ModHandle = getSPVRegister().registerSource(SPIRVModuleSpan);

  return (void **)ModHandle.Module;
}
```

**Called:** Once per compilation unit during static initialization

#### 2. __hipRegisterFunction

```cpp
extern "C" void __hipRegisterFunction(void **Data, const void *HostFunction,
                                      char *DeviceFunction,
                                      const char *FuncDeviceName, ...) {
  SPVRegister::Handle ModHandle{reinterpret_cast<void *>(Data)};
  getSPVRegister().bindFunction(ModHandle, HostPtr(HostFunction), FuncDeviceName);
}
```

**Called:** Once per `__global__` function during static initialization

#### 3. __hipRegisterVar

```cpp
extern "C" void __hipRegisterVar(void **Data, const void *HostVar,
                                 char *DeviceVar, const char *DeviceVarName,
                                 int Extern, size_t Size, int Constant, int Global) {
  SPVRegister::Handle ModHandle{reinterpret_cast<void *>(Data)};
  getSPVRegister().bindVariable(ModHandle, HostPtr(HostVar),
                                std::string(DeviceVarName), Size);
}
```

**Called:** Once per `__device__`/`__constant__` variable during static initialization

### Kernel Signature Communication

#### SPIR-V Type System

**Location:** `/home/yaakov/vortex_hip/chipStar/src/spirv.cc:525-1022`

The `SPIRVmodule` class parses SPIR-V instructions to build type information:

```cpp
class SPIRVmodule {
  std::map<InstWord, std::shared_ptr<SPIRVtype>> TypeMap_;
  std::map<InstWord, std::shared_ptr<SPIRVconstant>> ConstantMap_;
  std::map<InstWord, std::string_view> EntryPoints_;

  void fillModuleInfo(SPVModuleInfo &ModuleInfo);
};
```

**Type Inference:**
1. Parse `OpTypeInt`, `OpTypeFloat`, `OpTypeStruct` → `SPVTypeKind::POD`
2. Parse `OpTypePointer` → `SPVTypeKind::Pointer` + `StorageClass`
3. Parse `OpTypeImage` → `SPVTypeKind::Image`
4. Parse `OpTypeSampler` → `SPVTypeKind::Sampler`
5. Calculate sizes from type definitions

**Argument Extraction:**
1. Find `OpEntryPoint` instructions → kernel names
2. Lookup function type via `OpTypeFunction`
3. Iterate function parameters → argument types
4. Build `SPVArgTypeInfo` vector
5. Apply PODByRef annotations
6. Store in `SPVFuncInfo`

### SPVModuleInfo Structure

**Location:** `/home/yaakov/vortex_hip/chipStar/src/common.hh:46-52`

```cpp
struct SPVModuleInfo {
  SPVFunctionInfoMap FuncInfoMap;  // Kernel name → SPVFuncInfo
  bool HasNoIGBAs = false;         // Indirect global buffer access flag
};

using SPVFunctionInfoMap = std::map<std::string, std::shared_ptr<SPVFuncInfo>>;
```

**Populated by:** `analyzeSPIRV()` → `SPIRVmodule::fillModuleInfo()`

**Used by:**
- Backend compilation to create kernel wrappers
- Argument marshalling during kernel launch
- Kernel introspection (`hipFuncGetAttributes`)

### Compiler-Runtime Data Flow

```
HIP Source Code (.hip.cpp)
    ↓
Clang Frontend
    ↓
LLVM IR with HIP intrinsics
    ↓
chipStar LLVM Passes:
    - HipTextureLowering
    - HipKernelArgSpiller
    - HipDynMem
    - HipGlobalVariables
    - HipIGBADetector
    ↓
SPIR-V with annotations
    ↓
Clang offload bundler
    ↓
ELF binary with .hip_fatbin section
    ↓
Linked executable
    ↓
__hipRegisterFatBinary() [static init]
    ↓ Extract SPIR-V
SPVRegister::registerSource()
    ↓ Store OriginalBinary_
__hipRegisterFunction() [static init]
    ↓ Bind host ptr → kernel name
[LAZY] First kernel access
    ↓
analyzeSPIRV()
    ↓ Parse SPIR-V, extract types
fillModuleInfo()
    ↓ Build SPVModuleInfo with FuncInfoMap
Module::compile()
    ↓ Backend JIT compilation
Create Kernel wrappers with FuncInfo
    ↓
Ready for hipLaunchKernel()
```

### Key Design Principles

1. **Language Independence:** SPIR-V as common IR enables multiple frontends
2. **Metadata Preservation:** Type info flows from compiler to runtime unchanged
3. **Lazy Evaluation:** SPIR-V analysis deferred until needed
4. **Backend Abstraction:** Same metadata serves OpenCL and Level Zero
5. **Extensibility:** New argument types added via compiler passes and runtime handlers

---

## Summary

### Key File Paths Reference

**SPIR-V Processing:**
- `/home/yaakov/vortex_hip/chipStar/src/spirv.cc` (4881 lines) - SPIR-V parser and analyzer
- `/home/yaakov/vortex_hip/chipStar/src/spirv.hh` - SPIR-V definitions
- `/home/yaakov/vortex_hip/chipStar/tools/spirv-extractor/spirv-extractor.hh` - Binary extraction

**Metadata Structures:**
- `/home/yaakov/vortex_hip/chipStar/src/SPIRVFuncInfo.hh` - Kernel argument metadata
- `/home/yaakov/vortex_hip/chipStar/src/common.hh` - SPVModuleInfo definition

**Registration:**
- `/home/yaakov/vortex_hip/chipStar/src/SPVRegister.hh` - Global module registry interface
- `/home/yaakov/vortex_hip/chipStar/src/SPVRegister.cc` - Registration implementation
- `/home/yaakov/vortex_hip/chipStar/src/CHIPBindings.cc` (lines 6085-6247) - HIP API bindings

**Backend Architecture:**
- `/home/yaakov/vortex_hip/chipStar/src/CHIPBackend.hh` - Module and Kernel classes
- `/home/yaakov/vortex_hip/chipStar/src/CHIPBackend.cc` - Core implementations

**OpenCL Backend:**
- `/home/yaakov/vortex_hip/chipStar/src/backend/OpenCL/CHIPBackendOpenCL.hh`
- `/home/yaakov/vortex_hip/chipStar/src/backend/OpenCL/CHIPBackendOpenCL.cc` (line 2036) - Argument setup

**Level Zero Backend:**
- `/home/yaakov/vortex_hip/chipStar/src/backend/Level0/CHIPBackendLevel0.cc` (line 2964) - Argument setup

**Compiler Passes:**
- `/home/yaakov/vortex_hip/chipStar/llvm_passes/` - 35+ transformation passes

### Architecture Highlights

- **Lazy JIT Compilation:** SPIR-V finalization deferred until first kernel access
- **Visitor Pattern:** Flexible argument iteration for different use cases
- **Backend Abstraction:** Common SPIR-V pipeline serves multiple backends
- **Rich Metadata:** Complete type information preserved from compiler to runtime
- **Thread-Safe:** Global registry uses mutex protection
- **Memory Efficient:** Spill buffer reduces kernel parameter pressure
- **Extensible:** New argument types added without changing core architecture
