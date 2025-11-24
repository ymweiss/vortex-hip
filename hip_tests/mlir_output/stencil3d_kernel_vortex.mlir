module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, gpu.container_module, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", polygeist.gpu_module.llvm.data_layout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64", polygeist.gpu_module.llvm.target_triple = "nvptx64-nvidia-cuda", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  gpu.module @__polygeist_gpu_module {
    llvm.func @vx_get_threadIdx() -> !llvm.ptr
    llvm.func @vx_get_blockIdx() -> !llvm.ptr
    gpu.func @_Z16launch_stencil3dPfS_iii_kernel94833514235376(%arg0: index, %arg1: i32, %arg2: i32, %arg3: i1, %arg4: i32, %arg5: memref<?xf32>, %arg6: memref<?xf32>) kernel attributes {gpu.known_block_size = array<i32: 32, 1, 1>, nvvm.maxntidx = 32 : index, rocdl.max_flat_work_group_size = 32 : index} {
      %cst = arith.constant 0.000000e+00 : f32
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c-1 = arith.constant -1 : index
      %c3 = arith.constant 3 : index
      %c0_i32 = arith.constant 0 : i32
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
      %25 = arith.andi %24, %arg3 : i1
      %26 = arith.ori %23, %25 : i1
      %27 = arith.xori %26, %true : i1
      %28 = arith.andi %14, %27 : i1
      scf.if %28 {
        %29:2 = scf.for %arg7 = %c-1 to %c2 step %c1 iter_args(%arg8 = %c0_i32, %arg9 = %cst) -> (i32, f32) {
          %32 = arith.index_cast %arg7 : index to i32
          %33 = arith.cmpi slt, %32, %c0_i32 : i32
          %34 = scf.if %33 -> (i32) {
            scf.yield %c0_i32 : i32
          } else {
            %38 = arith.cmpi sge, %32, %arg2 : i32
            %39 = arith.select %38, %arg4, %32 : i32
            scf.yield %39 : i32
          }
          %35 = arith.muli %34, %arg2 : i32
          %36 = arith.muli %35, %arg2 : i32
          %37:2 = scf.for %arg10 = %c-1 to %c2 step %c1 iter_args(%arg11 = %arg8, %arg12 = %arg9) -> (i32, f32) {
            %38 = arith.index_cast %arg10 : index to i32
            %39 = arith.index_cast %arg11 : i32 to index
            %40 = arith.addi %39, %c3 : index
            %41 = arith.index_cast %40 : index to i32
            %42 = arith.cmpi slt, %38, %c0_i32 : i32
            %43 = scf.if %42 -> (i32) {
              scf.yield %c0_i32 : i32
            } else {
              %47 = arith.cmpi sge, %38, %arg2 : i32
              %48 = arith.select %47, %arg4, %38 : i32
              scf.yield %48 : i32
            }
            %44 = arith.muli %43, %arg2 : i32
            %45 = arith.addi %36, %44 : i32
            %46 = scf.for %arg13 = %c-1 to %c2 step %c1 iter_args(%arg14 = %arg12) -> (f32) {
              %47 = arith.index_cast %arg13 : index to i32
              %48 = arith.addi %18, %47 : i32
              %49 = arith.cmpi slt, %48, %c0_i32 : i32
              %50 = scf.if %49 -> (i32) {
                scf.yield %c0_i32 : i32
              } else {
                %55 = arith.cmpi sge, %48, %arg2 : i32
                %56 = arith.select %55, %arg4, %48 : i32
                scf.yield %56 : i32
              }
              %51 = arith.addi %45, %50 : i32
              %52 = arith.index_cast %51 : i32 to index
              %53 = memref.load %arg5[%52] : memref<?xf32>
              %54 = arith.addf %arg14, %53 : f32
              scf.yield %54 : f32
            }
            scf.yield %41, %46 : i32, f32
          }
          scf.yield %37#0, %37#1 : i32, f32
        }
        %30 = arith.sitofp %29#0 : i32 to f32
        %31 = arith.divf %29#1, %30 : f32
        memref.store %31, %arg6[%19] : memref<?xf32>
      }
      gpu.return
    }
  }
  func.func @_Z16launch_stencil3dPfS_iii(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: i32, %arg3: i32, %arg4: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c32 = arith.constant 32 : index
    %c0_i32 = arith.constant 0 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %arg3 : i32 to index
    %1 = arith.index_cast %arg4 : i32 to index
    %2 = arith.cmpi sle, %arg2, %c0_i32 : i32
    %3 = arith.addi %arg2, %c-1_i32 : i32
    %4 = arith.subi %1, %c1 : index
    %5 = arith.divui %4, %c32 : index
    %6 = arith.addi %5, %c1 : index
    %7 = "polygeist.gpu_error"() ({
      %8 = arith.cmpi sge, %0, %c1 : index
      %9 = arith.cmpi sge, %6, %c1 : index
      %10 = arith.andi %8, %9 : i1
      scf.if %10 {
        gpu.launch_func  @__polygeist_gpu_module::@_Z16launch_stencil3dPfS_iii_kernel94833514235376 blocks in (%0, %6, %c1) threads in (%c32, %c1, %c1)  args(%1 : index, %arg4 : i32, %arg2 : i32, %2 : i1, %3 : i32, %arg0 : memref<?xf32>, %arg1 : memref<?xf32>) {vortex.kernel_metadata = "Kernel: _Z16launch_stencil3dPfS_iii_kernel94833514235376\0ANum args: 7\0ATotal size (RV32): 28 bytes\0AArguments:\0A  [0] offset=0, size=4, type=scalar\0A  [1] offset=4, size=4, type=scalar\0A  [2] offset=8, size=4, type=scalar\0A  [3] offset=12, size=4, type=scalar\0A  [4] offset=16, size=4, type=scalar\0A  [5] offset=20, size=4, type=pointer\0A  [6] offset=24, size=4, type=pointer\0A"}
      }
      "polygeist.polygeist_yield"() : () -> ()
    }) : () -> index
    return
  }
}

