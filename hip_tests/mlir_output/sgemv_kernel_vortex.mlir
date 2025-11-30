module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, gpu.container_module, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", polygeist.gpu_module.llvm.data_layout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64", polygeist.gpu_module.llvm.target_triple = "nvptx64-nvidia-cuda", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  gpu.module @__polygeist_gpu_module {
    llvm.func @vx_get_threadIdx() -> !llvm.ptr
    llvm.func @vx_get_blockIdx() -> !llvm.ptr
    gpu.func @_Z12launch_sgemvPfS_S_jjii_kernel94204513471632(%arg0: index, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: memref<?xf32>, %arg5: memref<?xf32>, %arg6: index, %arg7: memref<?xf32>) kernel attributes {gpu.known_block_size = array<i32: 32, 1, 1>, nvvm.maxntidx = 32 : index, rocdl.max_flat_work_group_size = 32 : index} {
      %cst = arith.constant 0.000000e+00 : f32
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %c3 = arith.constant 3 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c32 = arith.constant 32 : index
      %0 = llvm.call @vx_get_blockIdx() : () -> !llvm.ptr
      %1 = llvm.getelementptr %0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32)>
      %2 = llvm.load %1 : !llvm.ptr -> i32
      %3 = builtin.unrealized_conversion_cast %2 : i32 to index
      %4 = llvm.call @vx_get_blockIdx() : () -> !llvm.ptr
      %5 = llvm.getelementptr %4[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32)>
      %6 = llvm.load %5 : !llvm.ptr -> i32
      %7 = builtin.unrealized_conversion_cast %6 : i32 to index
      %8 = llvm.call @vx_get_threadIdx() : () -> !llvm.ptr
      %9 = llvm.getelementptr %8[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32)>
      %10 = llvm.load %9 : !llvm.ptr -> i32
      %11 = builtin.unrealized_conversion_cast %10 : i32 to index
      %12 = arith.muli %7, %c32 : index
      %13 = arith.addi %12, %11 : index
      %14 = arith.cmpi ult, %13, %arg0 : index
      %15 = arith.index_cast %3 : index to i32
      %16 = arith.muli %15, %arg1 : i32
      %17 = arith.index_cast %13 : index to i32
      %18 = arith.addi %16, %17 : i32
      %19 = arith.index_cast %18 : i32 to index
      %20 = arith.cmpi slt, %18, %arg2 : i32
      %21 = arith.andi %14, %20 : i1
      scf.if %21 {
        %22 = arith.muli %18, %arg3 : i32
        %23 = scf.for %arg8 = %c0 to %arg6 step %c4 iter_args(%arg9 = %cst) -> (f32) {
          %24 = arith.index_cast %arg8 : index to i32
          %25 = arith.addi %22, %24 : i32
          %26 = arith.index_cast %25 : i32 to index
          %27 = memref.load %arg4[%26] : memref<?xf32>
          %28 = memref.load %arg5[%arg8] : memref<?xf32>
          %29 = arith.mulf %27, %28 : f32
          %30 = arith.addi %26, %c1 : index
          %31 = memref.load %arg4[%30] : memref<?xf32>
          %32 = arith.addi %arg8, %c1 : index
          %33 = memref.load %arg5[%32] : memref<?xf32>
          %34 = arith.mulf %31, %33 : f32
          %35 = arith.addf %29, %34 : f32
          %36 = arith.addi %26, %c2 : index
          %37 = memref.load %arg4[%36] : memref<?xf32>
          %38 = arith.addi %arg8, %c2 : index
          %39 = memref.load %arg5[%38] : memref<?xf32>
          %40 = arith.mulf %37, %39 : f32
          %41 = arith.addf %35, %40 : f32
          %42 = arith.addi %26, %c3 : index
          %43 = memref.load %arg4[%42] : memref<?xf32>
          %44 = arith.addi %arg8, %c3 : index
          %45 = memref.load %arg5[%44] : memref<?xf32>
          %46 = arith.mulf %43, %45 : f32
          %47 = arith.addf %41, %46 : f32
          %48 = arith.addf %arg9, %47 : f32
          scf.yield %48 : f32
        }
        memref.store %23, %arg7[%19] : memref<?xf32>
      }
      gpu.return
    }
  }
  func.func @_Z12launch_sgemvPfS_S_jjii(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %arg5 : i32 to index
    %1 = arith.index_cast %arg6 : i32 to index
    %2 = arith.index_cast %arg4 : i32 to index
    %3 = arith.subi %1, %c1 : index
    %4 = arith.divui %3, %c32 : index
    %5 = arith.addi %4, %c1 : index
    %6 = "polygeist.gpu_error"() ({
      %7 = arith.cmpi sge, %0, %c1 : index
      %8 = arith.cmpi sge, %5, %c1 : index
      %9 = arith.andi %7, %8 : i1
      scf.if %9 {
        gpu.launch_func  @__polygeist_gpu_module::@_Z12launch_sgemvPfS_S_jjii_kernel94204513471632 blocks in (%0, %5, %c1) threads in (%c32, %c1, %c1)  args(%1 : index, %arg6 : i32, %arg3 : i32, %arg4 : i32, %arg0 : memref<?xf32>, %arg1 : memref<?xf32>, %2 : index, %arg2 : memref<?xf32>) {vortex.kernel_metadata = "Kernel: _Z12launch_sgemvPfS_S_jjii_kernel94204513471632\0ANum args: 8\0ATotal size (RV32): 32 bytes\0AArguments:\0A  [0] offset=0, size=4, type=scalar\0A  [1] offset=4, size=4, type=scalar\0A  [2] offset=8, size=4, type=scalar\0A  [3] offset=12, size=4, type=scalar\0A  [4] offset=16, size=4, type=pointer\0A  [5] offset=20, size=4, type=pointer\0A  [6] offset=24, size=4, type=scalar\0A  [7] offset=28, size=4, type=pointer\0A"}
      }
      "polygeist.polygeist_yield"() : () -> ()
    }) : () -> index
    return
  }
}

