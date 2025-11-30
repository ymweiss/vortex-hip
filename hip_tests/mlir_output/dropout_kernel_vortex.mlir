module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, gpu.container_module, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", polygeist.gpu_module.llvm.data_layout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64", polygeist.gpu_module.llvm.target_triple = "nvptx64-nvidia-cuda", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  gpu.module @__polygeist_gpu_module {
    llvm.func @vx_get_threadIdx() -> !llvm.ptr
    llvm.func @vx_get_blockIdx() -> !llvm.ptr
    gpu.func @_Z14launch_dropoutPfS_ffjii_kernel94505661383840(%arg0: index, %arg1: i32, %arg2: i32, %arg3: memref<?xf32>, %arg4: f32, %arg5: f32, %arg6: memref<?xf32>) kernel attributes {gpu.known_block_size = array<i32: 32, 1, 1>, nvvm.maxntidx = 32 : index, rocdl.max_flat_work_group_size = 32 : index} {
      %cst = arith.constant 0.000000e+00 : f64
      %cst_0 = arith.constant 2.32830644E-10 : f32
      %c5_i32 = arith.constant 5 : i32
      %c17_i32 = arith.constant 17 : i32
      %c13_i32 = arith.constant 13 : i32
      %c15_i32 = arith.constant 15 : i32
      %c668265261_i32 = arith.constant 668265261 : i32
      %c4_i32 = arith.constant 4 : i32
      %c9_i32 = arith.constant 9 : i32
      %c16_i32 = arith.constant 16 : i32
      %c61_i32 = arith.constant 61 : i32
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
        %22 = arith.xori %18, %c61_i32 : i32
        %23 = arith.shrui %18, %c16_i32 : i32
        %24 = arith.xori %22, %23 : i32
        %25 = arith.muli %24, %c9_i32 : i32
        %26 = arith.shrui %25, %c4_i32 : i32
        %27 = arith.xori %25, %26 : i32
        %28 = arith.muli %27, %c668265261_i32 : i32
        %29 = arith.shrui %28, %c15_i32 : i32
        %30 = arith.xori %28, %29 : i32
        %31 = arith.shli %30, %c13_i32 : i32
        %32 = arith.xori %30, %31 : i32
        %33 = arith.shrui %32, %c17_i32 : i32
        %34 = arith.xori %32, %33 : i32
        %35 = arith.shli %34, %c5_i32 : i32
        %36 = arith.xori %34, %35 : i32
        %37 = arith.uitofp %36 : i32 to f32
        %38 = arith.mulf %37, %cst_0 : f32
        %39 = memref.load %arg3[%19] : memref<?xf32>
        %40 = arith.mulf %39, %arg4 : f32
        %41 = arith.cmpf olt, %38, %arg5 : f32
        %42 = scf.if %41 -> (f64) {
          scf.yield %cst : f64
        } else {
          %44 = arith.extf %40 : f32 to f64
          scf.yield %44 : f64
        }
        %43 = arith.truncf %42 : f64 to f32
        memref.store %43, %arg6[%19] : memref<?xf32>
      }
      gpu.return
    }
  }
  func.func @_Z14launch_dropoutPfS_ffjii(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: f32, %arg3: f32, %arg4: i32, %arg5: i32, %arg6: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %arg5 : i32 to index
    %1 = arith.index_cast %arg6 : i32 to index
    %2 = arith.subi %1, %c1 : index
    %3 = arith.divui %2, %c32 : index
    %4 = arith.addi %3, %c1 : index
    %5 = "polygeist.gpu_error"() ({
      %6 = arith.cmpi sge, %0, %c1 : index
      %7 = arith.cmpi sge, %4, %c1 : index
      %8 = arith.andi %6, %7 : i1
      scf.if %8 {
        gpu.launch_func  @__polygeist_gpu_module::@_Z14launch_dropoutPfS_ffjii_kernel94505661383840 blocks in (%0, %4, %c1) threads in (%c32, %c1, %c1)  args(%1 : index, %arg6 : i32, %arg4 : i32, %arg0 : memref<?xf32>, %arg3 : f32, %arg2 : f32, %arg1 : memref<?xf32>) {vortex.kernel_metadata = "Kernel: _Z14launch_dropoutPfS_ffjii_kernel94505661383840\0ANum args: 7\0ATotal size (RV32): 28 bytes\0AArguments:\0A  [0] offset=0, size=4, type=scalar\0A  [1] offset=4, size=4, type=scalar\0A  [2] offset=8, size=4, type=scalar\0A  [3] offset=12, size=4, type=pointer\0A  [4] offset=16, size=4, type=scalar\0A  [5] offset=20, size=4, type=scalar\0A  [6] offset=24, size=4, type=pointer\0A"}
      }
      "polygeist.polygeist_yield"() : () -> ()
    }) : () -> index
    return
  }
}

