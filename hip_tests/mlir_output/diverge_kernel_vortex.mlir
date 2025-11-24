module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, gpu.container_module, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", polygeist.gpu_module.llvm.data_layout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64", polygeist.gpu_module.llvm.target_triple = "nvptx64-nvidia-cuda", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  gpu.module @__polygeist_gpu_module {
    llvm.func @vx_get_threadIdx() -> !llvm.ptr
    llvm.func @vx_get_blockIdx() -> !llvm.ptr
    gpu.func @_Z14launch_divergePiS_jjii_kernel93862240521200(%arg0: index, %arg1: i32, %arg2: i32, %arg3: memref<?xi32>, %arg4: i32, %arg5: memref<?xi32>) kernel attributes {gpu.known_block_size = array<i32: 32, 1, 1>, nvvm.maxntidx = 32 : index, rocdl.max_flat_work_group_size = 32 : index} {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c7_i32 = arith.constant 7 : i32
      %c3_i32 = arith.constant 3 : i32
      %c4_i32 = arith.constant 4 : i32
      %c6_i32 = arith.constant 6 : i32
      %c2_i32 = arith.constant 2 : i32
      %c2147483647_i32 = arith.constant 2147483647 : i32
      %c-1_i32 = arith.constant -1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32
      %c5_i32 = arith.constant 5 : i32
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
      %20 = arith.cmpi slt, %18, %c5_i32 : i32
      %21 = arith.cmpi slt, %18, %arg2 : i32
      %22 = arith.andi %14, %21 : i1
      scf.if %22 {
        %23 = memref.load %arg3[%19] : memref<?xi32>
        %24 = arith.andi %18, %c1_i32 : i32
        %25 = arith.cmpi eq, %24, %c0_i32 : i32
        %26:2 = scf.while (%arg6 = %arg4, %arg7 = %23) : (i32, i32) -> (i32, i32) {
          %44 = arith.cmpi ne, %arg6, %c0_i32 : i32
          %45 = arith.andi %44, %25 : i1
          %46 = scf.if %45 -> (i32) {
            %47 = arith.addi %arg7, %c1_i32 : i32
            scf.yield %47 : i32
          } else {
            scf.yield %arg7 : i32
          }
          scf.condition(%44) %46, %arg6 : i32, i32
        } do {
        ^bb0(%arg6: i32, %arg7: i32):
          %44 = arith.addi %arg7, %c-1_i32 : i32
          scf.yield %44, %arg6 : i32, i32
        }
        %27 = arith.cmpi sge, %18, %c2147483647_i32 : i32
        %28 = scf.if %27 -> (i32) {
          scf.yield %c0_i32 : i32
        } else {
          %44 = arith.addi %26#0, %c2_i32 : i32
          scf.yield %44 : i32
        }
        %29 = arith.cmpi sgt, %18, %c1_i32 : i32
        %30 = scf.if %29 -> (i32) {
          %44 = arith.cmpi sgt, %18, %c2_i32 : i32
          %45 = scf.if %44 -> (i32) {
            %46 = arith.addi %28, %c6_i32 : i32
            scf.yield %46 : i32
          } else {
            %46 = arith.addi %28, %c5_i32 : i32
            scf.yield %46 : i32
          }
          scf.yield %45 : i32
        } else {
          %44 = arith.cmpi sgt, %18, %c0_i32 : i32
          %45 = scf.if %44 -> (i32) {
            %46 = arith.addi %28, %c4_i32 : i32
            scf.yield %46 : i32
          } else {
            %46 = arith.addi %28, %c3_i32 : i32
            scf.yield %46 : i32
          }
          scf.yield %45 : i32
        }
        %31 = arith.cmpi sge, %18, %c0_i32 : i32
        %32 = scf.if %31 -> (i32) {
          %44 = arith.addi %30, %c7_i32 : i32
          scf.yield %44 : i32
        } else {
          scf.yield %c0_i32 : i32
        }
        %33 = scf.for %arg6 = %c0 to %19 step %c1 iter_args(%arg7 = %32) -> (i32) {
          %44 = memref.load %arg3[%arg6] : memref<?xi32>
          %45 = arith.addi %arg7, %44 : i32
          scf.yield %45 : i32
        }
        %34 = scf.execute_region -> i32 {
          cf.switch %18 : i32, [
            default: ^bb4(%33 : i32),
            0: ^bb1,
            1: ^bb2(%33 : i32),
            2: ^bb3(%33, %c3_i32 : i32, i32),
            3: ^bb3(%33, %c5_i32 : i32, i32)
          ]
        ^bb1:  // pred: ^bb0
          %44 = arith.addi %33, %c1_i32 : i32
          cf.br ^bb4(%44 : i32)
        ^bb2(%45: i32):  // pred: ^bb0
          %46 = arith.addi %45, %c-1_i32 : i32
          cf.br ^bb4(%46 : i32)
        ^bb3(%47: i32, %48: i32):  // 2 preds: ^bb0, ^bb0
          %49 = arith.muli %47, %48 : i32
          cf.br ^bb4(%49 : i32)
        ^bb4(%50: i32):  // 4 preds: ^bb0, ^bb1, ^bb2, ^bb3
          scf.yield %50 : i32
        }
        %35 = scf.if %31 -> (i32) {
          %44 = arith.cmpi sgt, %18, %c5_i32 : i32
          %45 = scf.if %44 -> (i32) {
            %46 = memref.load %arg3[%c0] : memref<?xi32>
            scf.yield %46 : i32
          } else {
            scf.yield %18 : i32
          }
          scf.yield %45 : i32
        } else {
          %44 = scf.if %20 -> (i32) {
            %45 = memref.load %arg3[%c1] : memref<?xi32>
            scf.yield %45 : i32
          } else {
            %45 = arith.subi %c0_i32, %18 : i32
            scf.yield %45 : i32
          }
          scf.yield %44 : i32
        }
        %36 = arith.addi %34, %35 : i32
        %37 = memref.load %arg3[%19] : memref<?xi32>
        %38 = arith.cmpi slt, %37, %36 : i32
        %39 = arith.select %38, %37, %36 : i32
        %40 = arith.addi %36, %39 : i32
        %41 = arith.cmpi sgt, %37, %40 : i32
        %42 = arith.select %41, %37, %40 : i32
        %43 = arith.addi %40, %42 : i32
        memref.store %43, %arg5[%19] : memref<?xi32>
      }
      gpu.return
    }
  }
  func.func @_Z14launch_divergePiS_jjii(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %arg4 : i32 to index
    %1 = arith.index_cast %arg5 : i32 to index
    %2 = arith.subi %1, %c1 : index
    %3 = arith.divui %2, %c32 : index
    %4 = arith.addi %3, %c1 : index
    %5 = "polygeist.gpu_error"() ({
      %6 = arith.cmpi sge, %0, %c1 : index
      %7 = arith.cmpi sge, %4, %c1 : index
      %8 = arith.andi %6, %7 : i1
      scf.if %8 {
        gpu.launch_func  @__polygeist_gpu_module::@_Z14launch_divergePiS_jjii_kernel93862240521200 blocks in (%0, %4, %c1) threads in (%c32, %c1, %c1)  args(%1 : index, %arg5 : i32, %arg2 : i32, %arg0 : memref<?xi32>, %arg3 : i32, %arg1 : memref<?xi32>) {vortex.kernel_metadata = "Kernel: _Z14launch_divergePiS_jjii_kernel93862240521200\0ANum args: 6\0ATotal size (RV32): 24 bytes\0AArguments:\0A  [0] offset=0, size=4, type=scalar\0A  [1] offset=4, size=4, type=scalar\0A  [2] offset=8, size=4, type=scalar\0A  [3] offset=12, size=4, type=pointer\0A  [4] offset=16, size=4, type=scalar\0A  [5] offset=20, size=4, type=pointer\0A"}
      }
      "polygeist.polygeist_yield"() : () -> ()
    }) : () -> index
    return
  }
}

