module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, gpu.container_module, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", polygeist.gpu_module.llvm.data_layout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64", polygeist.gpu_module.llvm.target_triple = "nvptx64-nvidia-cuda", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  gpu.module @__polygeist_gpu_module {
    llvm.func @vx_get_threadIdx() -> !llvm.ptr
    llvm.func @vx_get_blockIdx() -> !llvm.ptr
    gpu.func @_Z12launch_fencePiS_S_jjii_kernel94493283566400(%arg0: index, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: memref<?xi32>, %arg5: memref<?xi32>, %arg6: memref<?xi32>, %arg7: index) kernel attributes {gpu.known_block_size = array<i32: 32, 1, 1>, nvvm.maxntidx = 32 : index, rocdl.max_flat_work_group_size = 32 : index} {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
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
      %19 = arith.cmpi slt, %18, %arg2 : i32
      %20 = arith.andi %14, %19 : i1
      scf.if %20 {
        %21 = arith.muli %18, %arg3 : i32
        scf.for %arg8 = %c0 to %arg7 step %c1 {
          %22 = arith.index_cast %arg8 : index to i32
          %23 = arith.addi %21, %22 : i32
          %24 = arith.index_cast %23 : i32 to index
          %25 = memref.load %arg4[%24] : memref<?xi32>
          %26 = memref.load %arg5[%24] : memref<?xi32>
          %27 = arith.addi %25, %26 : i32
          memref.store %27, %arg6[%24] : memref<?xi32>
        }
      }
      gpu.return
    }
  }
  func.func @_Z12launch_fencePiS_S_jjii(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: memref<?xi32>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %arg5 : i32 to index
    %1 = arith.index_cast %arg6 : i32 to index
    %2 = arith.index_cast %arg3 : i32 to index
    %3 = arith.subi %1, %c1 : index
    %4 = arith.divui %3, %c32 : index
    %5 = arith.addi %4, %c1 : index
    %6 = "polygeist.gpu_error"() ({
      %7 = arith.cmpi sge, %0, %c1 : index
      %8 = arith.cmpi sge, %5, %c1 : index
      %9 = arith.andi %7, %8 : i1
      scf.if %9 {
        gpu.launch_func  @__polygeist_gpu_module::@_Z12launch_fencePiS_S_jjii_kernel94493283566400 blocks in (%0, %5, %c1) threads in (%c32, %c1, %c1)  args(%1 : index, %arg6 : i32, %arg4 : i32, %arg3 : i32, %arg0 : memref<?xi32>, %arg1 : memref<?xi32>, %arg2 : memref<?xi32>, %2 : index) {vortex.kernel_metadata = "Kernel: _Z12launch_fencePiS_S_jjii_kernel94493283566400\0ANum args: 8\0ATotal size (RV32): 32 bytes\0AArguments:\0A  [0] offset=0, size=4, type=scalar\0A  [1] offset=4, size=4, type=scalar\0A  [2] offset=8, size=4, type=scalar\0A  [3] offset=12, size=4, type=scalar\0A  [4] offset=16, size=4, type=pointer\0A  [5] offset=20, size=4, type=pointer\0A  [6] offset=24, size=4, type=pointer\0A  [7] offset=28, size=4, type=scalar\0A"}
      }
      "polygeist.polygeist_yield"() : () -> ()
    }) : () -> index
    return
  }
}

