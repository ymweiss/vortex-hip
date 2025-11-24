module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, gpu.container_module, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", polygeist.gpu_module.llvm.data_layout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64", polygeist.gpu_module.llvm.target_triple = "nvptx64-nvidia-cuda", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  gpu.module @__polygeist_gpu_module {
    llvm.func @vx_printf(!llvm.ptr, ...) -> i32
    llvm.func @vx_core_id() -> i32
    llvm.func @vx_get_threadIdx() -> !llvm.ptr
    llvm.func @vx_get_blockIdx() -> !llvm.ptr
    gpu.func @_Z13launch_printfPcjii_kernel94614799239248(%arg0: index, %arg1: i32, %arg2: i32, %arg3: memref<?xi8>, %arg4: !llvm.ptr) kernel attributes {gpu.known_block_size = array<i32: 32, 1, 1>, nvvm.maxntidx = 32 : index, rocdl.max_flat_work_group_size = 32 : index} {
      %c65_i32 = arith.constant 65 : i32
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
        %22 = memref.load %arg3[%19] : memref<?xi8>
        %23 = arith.extsi %22 : i8 to i32
        %24 = arith.addi %23, %c65_i32 : i32
        %25 = arith.trunci %24 : i32 to i8
        %26 = arith.extsi %25 : i8 to i32
        %27 = llvm.call @vx_core_id() : () -> i32
        %28 = llvm.call @vx_printf(%arg4, %27, %18, %26) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32, i32, i32) -> i32
      }
      gpu.return
    }
    llvm.func @printf(!llvm.ptr, ...) -> i32
  }
  llvm.mlir.global internal constant @str0("task=%d, value=%c\0A\00") {addr_space = 0 : i32}
  llvm.func @printf(!llvm.ptr, ...) -> i32
  func.func @_Z13launch_printfPcjii(%arg0: memref<?xi8>, %arg1: i32, %arg2: i32, %arg3: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %arg2 : i32 to index
    %1 = arith.index_cast %arg3 : i32 to index
    %2 = llvm.mlir.addressof @str0 : !llvm.ptr
    %3 = llvm.getelementptr %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<19 x i8>
    %4 = arith.subi %1, %c1 : index
    %5 = arith.divui %4, %c32 : index
    %6 = arith.addi %5, %c1 : index
    %7 = "polygeist.gpu_error"() ({
      %8 = arith.cmpi sge, %0, %c1 : index
      %9 = arith.cmpi sge, %6, %c1 : index
      %10 = arith.andi %8, %9 : i1
      scf.if %10 {
        gpu.launch_func  @__polygeist_gpu_module::@_Z13launch_printfPcjii_kernel94614799239248 blocks in (%0, %6, %c1) threads in (%c32, %c1, %c1)  args(%1 : index, %arg3 : i32, %arg1 : i32, %arg0 : memref<?xi8>, %3 : !llvm.ptr) {vortex.kernel_metadata = "Kernel: _Z13launch_printfPcjii_kernel94614799239248\0ANum args: 5\0ATotal size (RV32): 20 bytes\0AArguments:\0A  [0] offset=0, size=4, type=scalar\0A  [1] offset=4, size=4, type=scalar\0A  [2] offset=8, size=4, type=scalar\0A  [3] offset=12, size=4, type=pointer\0A  [4] offset=16, size=4, type=scalar\0A"}
      }
      "polygeist.polygeist_yield"() : () -> ()
    }) : () -> index
    return
  }
}

