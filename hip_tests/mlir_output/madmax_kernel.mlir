module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, gpu.container_module, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", polygeist.gpu_module.llvm.data_layout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64", polygeist.gpu_module.llvm.target_triple = "nvptx64-nvidia-cuda", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  gpu.module @__polygeist_gpu_module {
    gpu.func @_Z13launch_madmaxPfiii_kernel94596009787536(%arg0: index, %arg1: i32, %arg2: i32, %arg3: i1, %arg4: memref<?xf32>) kernel attributes {gpu.known_block_size = array<i32: 32, 1, 1>, nvvm.maxntidx = 32 : index, rocdl.max_flat_work_group_size = 32 : index} {
      %c1 = arith.constant 1 : index
      %c256 = arith.constant 256 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 5.000000e-01 : f32
      %true = arith.constant true
      %c32 = arith.constant 32 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %c32 : index
      %4 = arith.addi %3, %2 : index
      %5 = arith.cmpi ult, %4, %arg0 : index
      %6 = arith.index_cast %0 : index to i32
      %7 = arith.muli %6, %arg1 : i32
      %8 = arith.index_cast %4 : index to i32
      %9 = arith.addi %7, %8 : i32
      %10 = arith.index_cast %9 : i32 to index
      %11 = arith.cmpi sge, %9, %arg2 : i32
      %12 = arith.cmpi slt, %9, %arg2 : i32
      %13 = arith.andi %12, %arg3 : i1
      %14 = arith.ori %11, %13 : i1
      %15 = arith.xori %14, %true : i1
      %16 = arith.andi %5, %15 : i1
      scf.if %16 {
        %17 = arith.sitofp %9 : i32 to f32
        %18 = arith.mulf %17, %cst : f32
        %19 = arith.muli %9, %arg2 : i32
        %20 = arith.sitofp %19 : i32 to f32
        %21 = arith.mulf %20, %cst : f32
        %22 = arith.addf %18, %21 : f32
        %23 = arith.subf %18, %21 : f32
        %24 = arith.mulf %22, %cst : f32
        %25 = arith.mulf %23, %cst : f32
        %26 = arith.addf %24, %25 : f32
        %27 = arith.subf %24, %25 : f32
        %28 = arith.mulf %26, %cst : f32
        %29 = arith.mulf %27, %cst : f32
        %30 = arith.addf %28, %29 : f32
        %31 = arith.subf %28, %29 : f32
        %32 = arith.mulf %30, %cst : f32
        %33 = arith.mulf %31, %cst : f32
        %34 = arith.addf %32, %33 : f32
        %35 = arith.subf %32, %33 : f32
        %36:16 = scf.for %arg5 = %c0 to %c256 step %c1 iter_args(%arg6 = %35, %arg7 = %34, %arg8 = %33, %arg9 = %32, %arg10 = %31, %arg11 = %30, %arg12 = %29, %arg13 = %28, %arg14 = %27, %arg15 = %26, %arg16 = %25, %arg17 = %24, %arg18 = %23, %arg19 = %22, %arg20 = %21, %arg21 = %18) -> (f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32) {
          %52 = arith.mulf %arg21, %arg20 : f32
          %53 = arith.addf %52, %arg19 : f32
          %54 = arith.mulf %arg20, %arg19 : f32
          %55 = arith.addf %54, %arg18 : f32
          %56 = arith.mulf %arg19, %arg18 : f32
          %57 = arith.addf %56, %arg17 : f32
          %58 = arith.mulf %arg18, %arg17 : f32
          %59 = arith.addf %58, %arg16 : f32
          %60 = arith.mulf %arg17, %arg16 : f32
          %61 = arith.addf %60, %arg15 : f32
          %62 = arith.mulf %arg16, %arg15 : f32
          %63 = arith.addf %62, %arg14 : f32
          %64 = arith.mulf %arg15, %arg14 : f32
          %65 = arith.addf %64, %arg13 : f32
          %66 = arith.mulf %arg14, %arg13 : f32
          %67 = arith.addf %66, %arg12 : f32
          %68 = arith.mulf %arg13, %arg12 : f32
          %69 = arith.addf %68, %arg11 : f32
          %70 = arith.mulf %arg12, %arg11 : f32
          %71 = arith.addf %70, %arg10 : f32
          %72 = arith.mulf %arg11, %arg10 : f32
          %73 = arith.addf %72, %arg9 : f32
          %74 = arith.mulf %arg10, %arg9 : f32
          %75 = arith.addf %74, %arg8 : f32
          %76 = arith.mulf %arg9, %arg8 : f32
          %77 = arith.addf %76, %arg7 : f32
          %78 = arith.mulf %arg8, %arg7 : f32
          %79 = arith.addf %78, %arg6 : f32
          %80 = arith.mulf %arg7, %arg6 : f32
          %81 = arith.addf %80, %53 : f32
          %82 = arith.mulf %arg6, %53 : f32
          %83 = arith.addf %82, %55 : f32
          scf.yield %83, %81, %79, %77, %75, %73, %71, %69, %67, %65, %63, %61, %59, %57, %55, %53 : f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32
        }
        %37 = arith.addf %36#15, %36#14 : f32
        %38 = arith.addf %37, %36#13 : f32
        %39 = arith.addf %38, %36#12 : f32
        %40 = arith.addf %39, %36#11 : f32
        %41 = arith.addf %40, %36#10 : f32
        %42 = arith.addf %41, %36#9 : f32
        %43 = arith.addf %42, %36#8 : f32
        %44 = arith.addf %43, %36#7 : f32
        %45 = arith.addf %44, %36#6 : f32
        %46 = arith.addf %45, %36#5 : f32
        %47 = arith.addf %46, %36#4 : f32
        %48 = arith.addf %47, %36#3 : f32
        %49 = arith.addf %48, %36#2 : f32
        %50 = arith.addf %49, %36#1 : f32
        %51 = arith.addf %50, %36#0 : f32
        memref.store %51, %arg4[%10] : memref<?xf32>
      }
      gpu.return
    }
  }
  func.func @_Z13launch_madmaxPfiii(%arg0: memref<?xf32>, %arg1: i32, %arg2: i32, %arg3: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c512 = arith.constant 512 : index
    %c1024 = arith.constant 1024 : index
    %c0_i32 = arith.constant 0 : i32
    %c256 = arith.constant 256 : index
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %arg2 : i32 to index
    %1 = arith.index_cast %arg3 : i32 to index
    %2 = arith.cmpi sle, %arg1, %c0_i32 : i32
    "polygeist.alternatives"() ({
      %3 = arith.subi %1, %c1 : index
      %4 = arith.divui %3, %c32 : index
      %5 = arith.addi %4, %c1 : index
      %6 = "polygeist.gpu_error"() ({
        %7 = arith.cmpi sge, %0, %c1 : index
        %8 = arith.cmpi sge, %5, %c1 : index
        %9 = arith.andi %7, %8 : i1
        scf.if %9 {
          gpu.launch_func  @__polygeist_gpu_module::@_Z13launch_madmaxPfiii_kernel94596009787536 blocks in (%0, %5, %c1) threads in (%c32, %c1, %c1)  args(%1 : index, %arg3 : i32, %arg1 : i32, %2 : i1, %arg0 : memref<?xf32>)
        }
        "polygeist.polygeist_yield"() : () -> ()
      }) : () -> index
      "polygeist.polygeist_yield"() : () -> ()
    }, {
      %3 = arith.subi %1, %c1 : index
      %4 = arith.divui %3, %c64 : index
      %5 = arith.addi %4, %c1 : index
      %6 = "polygeist.gpu_error"() ({
        %7 = arith.cmpi sge, %0, %c1 : index
        %8 = arith.cmpi sge, %5, %c1 : index
        %9 = arith.andi %7, %8 : i1
        scf.if %9 {
          gpu.launch_func  @__polygeist_gpu_module::@_Z13launch_madmaxPfiii_kernel94596009721040 blocks in (%0, %5, %c1) threads in (%c64, %c1, %c1)  args(%1 : index, %arg3 : i32, %arg1 : i32, %2 : i1, %arg0 : memref<?xf32>)
        }
        "polygeist.polygeist_yield"() : () -> ()
      }) : () -> index
      "polygeist.polygeist_yield"() : () -> ()
    }, {
      %3 = arith.subi %1, %c1 : index
      %4 = arith.divui %3, %c128 : index
      %5 = arith.addi %4, %c1 : index
      %6 = "polygeist.gpu_error"() ({
        %7 = arith.cmpi sge, %0, %c1 : index
        %8 = arith.cmpi sge, %5, %c1 : index
        %9 = arith.andi %7, %8 : i1
        scf.if %9 {
          gpu.launch_func  @__polygeist_gpu_module::@_Z13launch_madmaxPfiii_kernel94596009694656 blocks in (%0, %5, %c1) threads in (%c128, %c1, %c1)  args(%1 : index, %arg3 : i32, %arg1 : i32, %2 : i1, %arg0 : memref<?xf32>)
        }
        "polygeist.polygeist_yield"() : () -> ()
      }) : () -> index
      "polygeist.polygeist_yield"() : () -> ()
    }, {
      %3 = arith.subi %1, %c1 : index
      %4 = arith.divui %3, %c256 : index
      %5 = arith.addi %4, %c1 : index
      %6 = "polygeist.gpu_error"() ({
        %7 = arith.cmpi sge, %0, %c1 : index
        %8 = arith.cmpi sge, %5, %c1 : index
        %9 = arith.andi %7, %8 : i1
        scf.if %9 {
          gpu.launch_func  @__polygeist_gpu_module::@_Z13launch_madmaxPfiii_kernel94596009718832 blocks in (%0, %5, %c1) threads in (%c256, %c1, %c1)  args(%1 : index, %arg3 : i32, %arg1 : i32, %2 : i1, %arg0 : memref<?xf32>)
        }
        "polygeist.polygeist_yield"() : () -> ()
      }) : () -> index
      "polygeist.polygeist_yield"() : () -> ()
    }, {
      %3 = arith.subi %1, %c1 : index
      %4 = arith.divui %3, %c512 : index
      %5 = arith.addi %4, %c1 : index
      %6 = "polygeist.gpu_error"() ({
        %7 = arith.cmpi sge, %0, %c1 : index
        %8 = arith.cmpi sge, %5, %c1 : index
        %9 = arith.andi %7, %8 : i1
        scf.if %9 {
          gpu.launch_func  @__polygeist_gpu_module::@_Z13launch_madmaxPfiii_kernel94596009770368 blocks in (%0, %5, %c1) threads in (%c512, %c1, %c1)  args(%1 : index, %arg3 : i32, %arg1 : i32, %2 : i1, %arg0 : memref<?xf32>)
        }
        "polygeist.polygeist_yield"() : () -> ()
      }) : () -> index
      "polygeist.polygeist_yield"() : () -> ()
    }, {
      %3 = arith.subi %1, %c1 : index
      %4 = arith.divui %3, %c1024 : index
      %5 = arith.addi %4, %c1 : index
      %6 = "polygeist.gpu_error"() ({
        %7 = arith.cmpi sge, %0, %c1 : index
        %8 = arith.cmpi sge, %5, %c1 : index
        %9 = arith.andi %7, %8 : i1
        scf.if %9 {
          gpu.launch_func  @__polygeist_gpu_module::@_Z13launch_madmaxPfiii_kernel94596009794480 blocks in (%0, %5, %c1) threads in (%c1024, %c1, %c1)  args(%1 : index, %arg3 : i32, %arg1 : i32, %2 : i1, %arg0 : memref<?xf32>)
        }
        "polygeist.polygeist_yield"() : () -> ()
      }) : () -> index
      "polygeist.polygeist_yield"() : () -> ()
    }) {alternatives.descs = ["block_size=32,blockDims=x:32;y:1;z:1;,floatOps=32:263136;,intOps=4:96;8:64;,loads=,stores=4/x:unk|y:unk|z:unk|/0:32;,", "block_size=64,blockDims=x:64;y:1;z:1;,floatOps=32:526272;,intOps=4:192;8:128;,loads=,stores=4/x:unk|y:unk|z:unk|/0:64;,", "block_size=128,blockDims=x:128;y:1;z:1;,floatOps=32:1052544;,intOps=4:384;8:256;,loads=,stores=4/x:unk|y:unk|z:unk|/0:128;,", "block_size=256,blockDims=x:256;y:1;z:1;,floatOps=32:2105088;,intOps=4:768;8:512;,loads=,stores=4/x:unk|y:unk|z:unk|/0:256;,", "block_size=512,blockDims=x:512;y:1;z:1;,floatOps=32:4210176;,intOps=4:1536;8:1024;,loads=,stores=4/x:unk|y:unk|z:unk|/0:512;,", "block_size=1024,blockDims=x:1024;y:1;z:1;,floatOps=32:8420352;,intOps=4:3072;8:2048;,loads=,stores=4/x:unk|y:unk|z:unk|/0:1024;,"], alternatives.type = "gpu_kernel", polygeist.altop.id = "+home+yaakov+vortex_hiploc(\22+tmp+polygeist_temp_madmax_kernel.cu\22:56:3)_Z13launch_madmaxPfiii.func.0"} : () -> ()
    return
  }
}
