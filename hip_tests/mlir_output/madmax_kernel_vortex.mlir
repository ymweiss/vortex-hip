module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, gpu.container_module, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", polygeist.gpu_module.llvm.data_layout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64", polygeist.gpu_module.llvm.target_triple = "nvptx64-nvidia-cuda", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  gpu.module @__polygeist_gpu_module {
    llvm.func @vx_get_threadIdx() -> !llvm.ptr
    llvm.func @vx_get_blockIdx() -> !llvm.ptr
    gpu.func @_Z13launch_madmaxPfiii_kernel94596009787536(%arg0: index, %arg1: i32, %arg2: i32, %arg3: i1, %arg4: memref<?xf32>) kernel attributes {gpu.known_block_size = array<i32: 32, 1, 1>, nvvm.maxntidx = 32 : index, rocdl.max_flat_work_group_size = 32 : index} {
      %c1 = arith.constant 1 : index
      %c256 = arith.constant 256 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 5.000000e-01 : f32
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
        %26 = arith.sitofp %18 : i32 to f32
        %27 = arith.mulf %26, %cst : f32
        %28 = arith.muli %18, %arg2 : i32
        %29 = arith.sitofp %28 : i32 to f32
        %30 = arith.mulf %29, %cst : f32
        %31 = arith.addf %27, %30 : f32
        %32 = arith.subf %27, %30 : f32
        %33 = arith.mulf %31, %cst : f32
        %34 = arith.mulf %32, %cst : f32
        %35 = arith.addf %33, %34 : f32
        %36 = arith.subf %33, %34 : f32
        %37 = arith.mulf %35, %cst : f32
        %38 = arith.mulf %36, %cst : f32
        %39 = arith.addf %37, %38 : f32
        %40 = arith.subf %37, %38 : f32
        %41 = arith.mulf %39, %cst : f32
        %42 = arith.mulf %40, %cst : f32
        %43 = arith.addf %41, %42 : f32
        %44 = arith.subf %41, %42 : f32
        %45:16 = scf.for %arg5 = %c0 to %c256 step %c1 iter_args(%arg6 = %44, %arg7 = %43, %arg8 = %42, %arg9 = %41, %arg10 = %40, %arg11 = %39, %arg12 = %38, %arg13 = %37, %arg14 = %36, %arg15 = %35, %arg16 = %34, %arg17 = %33, %arg18 = %32, %arg19 = %31, %arg20 = %30, %arg21 = %27) -> (f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32) {
          %61 = arith.mulf %arg21, %arg20 : f32
          %62 = arith.addf %61, %arg19 : f32
          %63 = arith.mulf %arg20, %arg19 : f32
          %64 = arith.addf %63, %arg18 : f32
          %65 = arith.mulf %arg19, %arg18 : f32
          %66 = arith.addf %65, %arg17 : f32
          %67 = arith.mulf %arg18, %arg17 : f32
          %68 = arith.addf %67, %arg16 : f32
          %69 = arith.mulf %arg17, %arg16 : f32
          %70 = arith.addf %69, %arg15 : f32
          %71 = arith.mulf %arg16, %arg15 : f32
          %72 = arith.addf %71, %arg14 : f32
          %73 = arith.mulf %arg15, %arg14 : f32
          %74 = arith.addf %73, %arg13 : f32
          %75 = arith.mulf %arg14, %arg13 : f32
          %76 = arith.addf %75, %arg12 : f32
          %77 = arith.mulf %arg13, %arg12 : f32
          %78 = arith.addf %77, %arg11 : f32
          %79 = arith.mulf %arg12, %arg11 : f32
          %80 = arith.addf %79, %arg10 : f32
          %81 = arith.mulf %arg11, %arg10 : f32
          %82 = arith.addf %81, %arg9 : f32
          %83 = arith.mulf %arg10, %arg9 : f32
          %84 = arith.addf %83, %arg8 : f32
          %85 = arith.mulf %arg9, %arg8 : f32
          %86 = arith.addf %85, %arg7 : f32
          %87 = arith.mulf %arg8, %arg7 : f32
          %88 = arith.addf %87, %arg6 : f32
          %89 = arith.mulf %arg7, %arg6 : f32
          %90 = arith.addf %89, %62 : f32
          %91 = arith.mulf %arg6, %62 : f32
          %92 = arith.addf %91, %64 : f32
          scf.yield %92, %90, %88, %86, %84, %82, %80, %78, %76, %74, %72, %70, %68, %66, %64, %62 : f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32
        }
        %46 = arith.addf %45#15, %45#14 : f32
        %47 = arith.addf %46, %45#13 : f32
        %48 = arith.addf %47, %45#12 : f32
        %49 = arith.addf %48, %45#11 : f32
        %50 = arith.addf %49, %45#10 : f32
        %51 = arith.addf %50, %45#9 : f32
        %52 = arith.addf %51, %45#8 : f32
        %53 = arith.addf %52, %45#7 : f32
        %54 = arith.addf %53, %45#6 : f32
        %55 = arith.addf %54, %45#5 : f32
        %56 = arith.addf %55, %45#4 : f32
        %57 = arith.addf %56, %45#3 : f32
        %58 = arith.addf %57, %45#2 : f32
        %59 = arith.addf %58, %45#1 : f32
        %60 = arith.addf %59, %45#0 : f32
        memref.store %60, %arg4[%19] : memref<?xf32>
      }
      gpu.return
    }
  }
  func.func @_Z13launch_madmaxPfiii(%arg0: memref<?xf32>, %arg1: i32, %arg2: i32, %arg3: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c32 = arith.constant 32 : index
    %c0_i32 = arith.constant 0 : i32
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %arg2 : i32 to index
    %1 = arith.index_cast %arg3 : i32 to index
    %2 = arith.cmpi sle, %arg1, %c0_i32 : i32
    %3 = arith.subi %1, %c1 : index
    %4 = arith.divui %3, %c32 : index
    %5 = arith.addi %4, %c1 : index
    %6 = "polygeist.gpu_error"() ({
      %7 = arith.cmpi sge, %0, %c1 : index
      %8 = arith.cmpi sge, %5, %c1 : index
      %9 = arith.andi %7, %8 : i1
      scf.if %9 {
        gpu.launch_func  @__polygeist_gpu_module::@_Z13launch_madmaxPfiii_kernel94596009787536 blocks in (%0, %5, %c1) threads in (%c32, %c1, %c1)  args(%1 : index, %arg3 : i32, %arg1 : i32, %2 : i1, %arg0 : memref<?xf32>) {vortex.kernel_metadata = "Kernel: _Z13launch_madmaxPfiii_kernel94596009787536\0ANum args: 5\0ATotal size (RV32): 20 bytes\0AArguments:\0A  [0] offset=0, size=4, type=scalar\0A  [1] offset=4, size=4, type=scalar\0A  [2] offset=8, size=4, type=scalar\0A  [3] offset=12, size=4, type=scalar\0A  [4] offset=16, size=4, type=pointer\0A"}
      }
      "polygeist.polygeist_yield"() : () -> ()
    }) : () -> index
    return
  }
}

