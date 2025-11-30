module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, gpu.container_module, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", polygeist.gpu_module.llvm.data_layout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64", polygeist.gpu_module.llvm.target_triple = "nvptx64-nvidia-cuda", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  gpu.module @__polygeist_gpu_module {
    llvm.func @vx_get_threadIdx() -> !llvm.ptr
    llvm.func @vx_get_blockIdx() -> !llvm.ptr
    gpu.func @_Z12launch_conv3PfS_S_iii_kernel94180481859392(%arg0: index, %arg1: i32, %arg2: i32, %arg3: i1, %arg4: memref<?xf32>, %arg5: memref<?xf32>, %arg6: index, %arg7: index, %arg8: memref<?xf32>) kernel attributes {gpu.known_block_size = array<i32: 32, 1, 1>, nvvm.maxntidx = 32 : index, rocdl.max_flat_work_group_size = 32 : index} {
      %c8 = arith.constant 8 : index
      %c7 = arith.constant 7 : index
      %c6 = arith.constant 6 : index
      %c5 = arith.constant 5 : index
      %c4 = arith.constant 4 : index
      %c3 = arith.constant 3 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %cst = arith.constant 0.000000e+00 : f32
      %c0 = arith.constant 0 : index
      %true = arith.constant true
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
      %20 = arith.cmpi sge, %18, %arg2 : i32
      %21 = arith.cmpi slt, %18, %arg2 : i32
      %22 = arith.andi %21, %arg3 : i1
      %23 = arith.ori %20, %22 : i1
      %24 = arith.xori %23, %true : i1
      %25 = arith.andi %14, %24 : i1
      scf.if %25 {
        %26 = memref.load %arg4[%19] : memref<?xf32>
        %27 = memref.load %arg5[%c0] : memref<?xf32>
        %28 = arith.mulf %26, %27 : f32
        %29 = arith.addf %28, %cst : f32
        %30 = arith.addi %19, %c1 : index
        %31 = memref.load %arg4[%30] : memref<?xf32>
        %32 = memref.load %arg5[%c1] : memref<?xf32>
        %33 = arith.mulf %31, %32 : f32
        %34 = arith.addf %29, %33 : f32
        %35 = arith.addi %19, %c2 : index
        %36 = memref.load %arg4[%35] : memref<?xf32>
        %37 = memref.load %arg5[%c2] : memref<?xf32>
        %38 = arith.mulf %36, %37 : f32
        %39 = arith.addf %34, %38 : f32
        %40 = arith.addi %arg6, %19 : index
        %41 = memref.load %arg4[%40] : memref<?xf32>
        %42 = memref.load %arg5[%c3] : memref<?xf32>
        %43 = arith.mulf %41, %42 : f32
        %44 = arith.addf %39, %43 : f32
        %45 = arith.addi %40, %c1 : index
        %46 = memref.load %arg4[%45] : memref<?xf32>
        %47 = memref.load %arg5[%c4] : memref<?xf32>
        %48 = arith.mulf %46, %47 : f32
        %49 = arith.addf %44, %48 : f32
        %50 = arith.addi %40, %c2 : index
        %51 = memref.load %arg4[%50] : memref<?xf32>
        %52 = memref.load %arg5[%c5] : memref<?xf32>
        %53 = arith.mulf %51, %52 : f32
        %54 = arith.addf %49, %53 : f32
        %55 = arith.addi %arg7, %19 : index
        %56 = memref.load %arg4[%55] : memref<?xf32>
        %57 = memref.load %arg5[%c6] : memref<?xf32>
        %58 = arith.mulf %56, %57 : f32
        %59 = arith.addf %54, %58 : f32
        %60 = arith.addi %55, %c1 : index
        %61 = memref.load %arg4[%60] : memref<?xf32>
        %62 = memref.load %arg5[%c7] : memref<?xf32>
        %63 = arith.mulf %61, %62 : f32
        %64 = arith.addf %59, %63 : f32
        %65 = arith.addi %55, %c2 : index
        %66 = memref.load %arg4[%65] : memref<?xf32>
        %67 = memref.load %arg5[%c8] : memref<?xf32>
        %68 = arith.mulf %66, %67 : f32
        %69 = arith.addf %64, %68 : f32
        memref.store %69, %arg8[%19] : memref<?xf32>
      }
      gpu.return
    }
  }
  func.func @_Z12launch_conv3PfS_S_iii(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: i32, %arg4: i32, %arg5: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c32 = arith.constant 32 : index
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %arg4 : i32 to index
    %1 = arith.index_cast %arg5 : i32 to index
    %2 = arith.addi %arg3, %c2_i32 : i32
    %3 = arith.index_cast %2 : i32 to index
    %4 = arith.muli %2, %c2_i32 : i32
    %5 = arith.index_cast %4 : i32 to index
    %6 = arith.cmpi sle, %arg3, %c0_i32 : i32
    %7 = arith.subi %1, %c1 : index
    %8 = arith.divui %7, %c32 : index
    %9 = arith.addi %8, %c1 : index
    %10 = "polygeist.gpu_error"() ({
      %11 = arith.cmpi sge, %0, %c1 : index
      %12 = arith.cmpi sge, %9, %c1 : index
      %13 = arith.andi %11, %12 : i1
      scf.if %13 {
        gpu.launch_func  @__polygeist_gpu_module::@_Z12launch_conv3PfS_S_iii_kernel94180481859392 blocks in (%0, %9, %c1) threads in (%c32, %c1, %c1)  args(%1 : index, %arg5 : i32, %arg3 : i32, %6 : i1, %arg0 : memref<?xf32>, %arg1 : memref<?xf32>, %3 : index, %5 : index, %arg2 : memref<?xf32>) {vortex.kernel_metadata = "Kernel: _Z12launch_conv3PfS_S_iii_kernel94180481859392\0ANum args: 9\0ATotal size (RV32): 36 bytes\0AArguments:\0A  [0] offset=0, size=4, type=scalar\0A  [1] offset=4, size=4, type=scalar\0A  [2] offset=8, size=4, type=scalar\0A  [3] offset=12, size=4, type=scalar\0A  [4] offset=16, size=4, type=pointer\0A  [5] offset=20, size=4, type=pointer\0A  [6] offset=24, size=4, type=scalar\0A  [7] offset=28, size=4, type=scalar\0A  [8] offset=32, size=4, type=pointer\0A"}
      }
      "polygeist.polygeist_yield"() : () -> ()
    }) : () -> index
    return
  }
}

