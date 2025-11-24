module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, gpu.container_module, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", polygeist.gpu_module.llvm.data_layout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64", polygeist.gpu_module.llvm.target_triple = "nvptx64-nvidia-cuda", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  gpu.module @__polygeist_gpu_module {
    memref.global @shared_mem_93991622930752 : memref<1xf32, 3> = uninitialized
    gpu.func @_Z13launch_sgemm2PfS_S_iiii_kernel93991622908448(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: index, %arg4: index, %arg5: memref<?xf32>, %arg6: i1, %arg7: memref<?xf32>, %arg8: index, %arg9: memref<?xf32>) kernel {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %false = arith.constant false
      %0 = gpu.block_id  x
      %1 = memref.get_global @shared_mem_93991622930752 : memref<1xf32, 3>
      %2 = gpu.thread_id  x
      %3 = arith.index_cast %0 : index to i32
      %4 = arith.muli %3, %arg0 : i32
      %5 = arith.index_cast %2 : index to i32
      %6 = arith.muli %5, %arg1 : i32
      %7 = arith.index_cast %6 : i32 to index
      %8 = arith.addi %4, %5 : i32
      %9 = arith.muli %8, %arg2 : i32
      %10 = arith.index_cast %9 : i32 to index
      %11 = arith.cmpi slt, %8, %arg2 : i32
      %12 = arith.addi %7, %arg3 : index
      %13 = scf.for %arg10 = %c0 to %arg8 step %arg4 iter_args(%arg11 = %cst) -> (f32) {
        %15 = arith.divui %arg10, %arg4 : index
        %16 = arith.muli %15, %arg4 : index
        %17 = arith.index_cast %16 : index to i32
        %18 = scf.if %11 -> (i1) {
          %23 = arith.cmpi slt, %17, %arg2 : i32
          scf.yield %23 : i1
        } else {
          scf.yield %false : i1
        }
        scf.if %18 {
          %23 = arith.addi %9, %17 : i32
          %24 = arith.index_cast %23 : i32 to index
          %25 = memref.load %arg5[%24] : memref<?xf32>
          memref.store %25, %1[%7] : memref<1xf32, 3>
        } else {
          memref.store %cst, %1[%7] : memref<1xf32, 3>
        }
        %19 = arith.addi %17, %5 : i32
        %20 = arith.cmpi slt, %19, %arg2 : i32
        %21 = arith.andi %20, %arg6 : i1
        scf.if %21 {
          %23 = arith.muli %19, %arg2 : i32
          %24 = arith.index_cast %23 : i32 to index
          %25 = memref.load %arg7[%24] : memref<?xf32>
          memref.store %25, %1[%12] : memref<1xf32, 3>
        } else {
          memref.store %cst, %1[%12] : memref<1xf32, 3>
        }
        gpu.barrier
        %22 = scf.for %arg12 = %c0 to %arg4 step %c1 iter_args(%arg13 = %arg11) -> (f32) {
          %23 = arith.index_cast %arg12 : index to i32
          %24 = arith.addi %6, %23 : i32
          %25 = arith.index_cast %24 : i32 to index
          %26 = memref.load %1[%25] : memref<1xf32, 3>
          %27 = arith.muli %23, %arg1 : i32
          %28 = arith.index_cast %27 : i32 to index
          %29 = arith.addi %28, %arg3 : index
          %30 = memref.load %1[%29] : memref<1xf32, 3>
          %31 = arith.mulf %26, %30 : f32
          %32 = arith.addf %arg13, %31 : f32
          scf.yield %32 : f32
        }
        gpu.barrier
        scf.yield %22 : f32
      }
      %14 = arith.andi %11, %arg6 : i1
      scf.if %14 {
        memref.store %13, %arg9[%10] : memref<?xf32>
      }
      gpu.return
    }
  }
  func.func @_Z13launch_sgemm2PfS_S_iiii(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.index_cast %arg5 : i32 to index
    %1 = arith.index_cast %arg6 : i32 to index
    %2 = arith.muli %arg4, %arg4 : i32
    %3 = arith.index_cast %2 : i32 to index
    %4 = arith.index_cast %arg3 : i32 to index
    %5 = arith.index_cast %arg4 : i32 to index
    %6 = arith.cmpi sgt, %arg3, %c0_i32 : i32
    %7 = "polygeist.gpu_error"() ({
      %8 = arith.cmpi sge, %0, %c1 : index
      scf.if %8 {
        gpu.launch_func  @__polygeist_gpu_module::@_Z13launch_sgemm2PfS_S_iiii_kernel93991622908448 blocks in (%0, %c1, %c1) threads in (%1, %c1, %c1)  args(%arg6 : i32, %arg4 : i32, %arg3 : i32, %3 : index, %5 : index, %arg0 : memref<?xf32>, %6 : i1, %arg1 : memref<?xf32>, %4 : index, %arg2 : memref<?xf32>)
      }
      "polygeist.polygeist_yield"() : () -> ()
    }) : () -> index
    return
  }
}
