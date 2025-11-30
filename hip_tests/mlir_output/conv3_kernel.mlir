module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, gpu.container_module, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", polygeist.gpu_module.llvm.data_layout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64", polygeist.gpu_module.llvm.target_triple = "nvptx64-nvidia-cuda", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  gpu.module @__polygeist_gpu_module {
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
        %17 = memref.load %arg4[%10] : memref<?xf32>
        %18 = memref.load %arg5[%c0] : memref<?xf32>
        %19 = arith.mulf %17, %18 : f32
        %20 = arith.addf %19, %cst : f32
        %21 = arith.addi %10, %c1 : index
        %22 = memref.load %arg4[%21] : memref<?xf32>
        %23 = memref.load %arg5[%c1] : memref<?xf32>
        %24 = arith.mulf %22, %23 : f32
        %25 = arith.addf %20, %24 : f32
        %26 = arith.addi %10, %c2 : index
        %27 = memref.load %arg4[%26] : memref<?xf32>
        %28 = memref.load %arg5[%c2] : memref<?xf32>
        %29 = arith.mulf %27, %28 : f32
        %30 = arith.addf %25, %29 : f32
        %31 = arith.addi %arg6, %10 : index
        %32 = memref.load %arg4[%31] : memref<?xf32>
        %33 = memref.load %arg5[%c3] : memref<?xf32>
        %34 = arith.mulf %32, %33 : f32
        %35 = arith.addf %30, %34 : f32
        %36 = arith.addi %31, %c1 : index
        %37 = memref.load %arg4[%36] : memref<?xf32>
        %38 = memref.load %arg5[%c4] : memref<?xf32>
        %39 = arith.mulf %37, %38 : f32
        %40 = arith.addf %35, %39 : f32
        %41 = arith.addi %31, %c2 : index
        %42 = memref.load %arg4[%41] : memref<?xf32>
        %43 = memref.load %arg5[%c5] : memref<?xf32>
        %44 = arith.mulf %42, %43 : f32
        %45 = arith.addf %40, %44 : f32
        %46 = arith.addi %arg7, %10 : index
        %47 = memref.load %arg4[%46] : memref<?xf32>
        %48 = memref.load %arg5[%c6] : memref<?xf32>
        %49 = arith.mulf %47, %48 : f32
        %50 = arith.addf %45, %49 : f32
        %51 = arith.addi %46, %c1 : index
        %52 = memref.load %arg4[%51] : memref<?xf32>
        %53 = memref.load %arg5[%c7] : memref<?xf32>
        %54 = arith.mulf %52, %53 : f32
        %55 = arith.addf %50, %54 : f32
        %56 = arith.addi %46, %c2 : index
        %57 = memref.load %arg4[%56] : memref<?xf32>
        %58 = memref.load %arg5[%c8] : memref<?xf32>
        %59 = arith.mulf %57, %58 : f32
        %60 = arith.addf %55, %59 : f32
        memref.store %60, %arg8[%10] : memref<?xf32>
      }
      gpu.return
    }
  }
  func.func @_Z12launch_conv3PfS_S_iii(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: i32, %arg4: i32, %arg5: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c512 = arith.constant 512 : index
    %c1024 = arith.constant 1024 : index
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
    "polygeist.alternatives"() ({
      %7 = arith.subi %1, %c1 : index
      %8 = arith.divui %7, %c32 : index
      %9 = arith.addi %8, %c1 : index
      %10 = "polygeist.gpu_error"() ({
        %11 = arith.cmpi sge, %0, %c1 : index
        %12 = arith.cmpi sge, %9, %c1 : index
        %13 = arith.andi %11, %12 : i1
        scf.if %13 {
          gpu.launch_func  @__polygeist_gpu_module::@_Z12launch_conv3PfS_S_iii_kernel94180481859392 blocks in (%0, %9, %c1) threads in (%c32, %c1, %c1)  args(%1 : index, %arg5 : i32, %arg3 : i32, %6 : i1, %arg0 : memref<?xf32>, %arg1 : memref<?xf32>, %3 : index, %5 : index, %arg2 : memref<?xf32>)
        }
        "polygeist.polygeist_yield"() : () -> ()
      }) : () -> index
      "polygeist.polygeist_yield"() : () -> ()
    }, {
      %7 = arith.subi %1, %c1 : index
      %8 = arith.divui %7, %c64 : index
      %9 = arith.addi %8, %c1 : index
      %10 = "polygeist.gpu_error"() ({
        %11 = arith.cmpi sge, %0, %c1 : index
        %12 = arith.cmpi sge, %9, %c1 : index
        %13 = arith.andi %11, %12 : i1
        scf.if %13 {
          gpu.launch_func  @__polygeist_gpu_module::@_Z12launch_conv3PfS_S_iii_kernel94180483938064 blocks in (%0, %9, %c1) threads in (%c64, %c1, %c1)  args(%1 : index, %arg5 : i32, %arg3 : i32, %6 : i1, %arg0 : memref<?xf32>, %arg1 : memref<?xf32>, %3 : index, %5 : index, %arg2 : memref<?xf32>)
        }
        "polygeist.polygeist_yield"() : () -> ()
      }) : () -> index
      "polygeist.polygeist_yield"() : () -> ()
    }, {
      %7 = arith.subi %1, %c1 : index
      %8 = arith.divui %7, %c128 : index
      %9 = arith.addi %8, %c1 : index
      %10 = "polygeist.gpu_error"() ({
        %11 = arith.cmpi sge, %0, %c1 : index
        %12 = arith.cmpi sge, %9, %c1 : index
        %13 = arith.andi %11, %12 : i1
        scf.if %13 {
          gpu.launch_func  @__polygeist_gpu_module::@_Z12launch_conv3PfS_S_iii_kernel94180483954672 blocks in (%0, %9, %c1) threads in (%c128, %c1, %c1)  args(%1 : index, %arg5 : i32, %arg3 : i32, %6 : i1, %arg0 : memref<?xf32>, %arg1 : memref<?xf32>, %3 : index, %5 : index, %arg2 : memref<?xf32>)
        }
        "polygeist.polygeist_yield"() : () -> ()
      }) : () -> index
      "polygeist.polygeist_yield"() : () -> ()
    }, {
      %7 = arith.subi %1, %c1 : index
      %8 = arith.divui %7, %c256 : index
      %9 = arith.addi %8, %c1 : index
      %10 = "polygeist.gpu_error"() ({
        %11 = arith.cmpi sge, %0, %c1 : index
        %12 = arith.cmpi sge, %9, %c1 : index
        %13 = arith.andi %11, %12 : i1
        scf.if %13 {
          gpu.launch_func  @__polygeist_gpu_module::@_Z12launch_conv3PfS_S_iii_kernel94180483973440 blocks in (%0, %9, %c1) threads in (%c256, %c1, %c1)  args(%1 : index, %arg5 : i32, %arg3 : i32, %6 : i1, %arg0 : memref<?xf32>, %arg1 : memref<?xf32>, %3 : index, %5 : index, %arg2 : memref<?xf32>)
        }
        "polygeist.polygeist_yield"() : () -> ()
      }) : () -> index
      "polygeist.polygeist_yield"() : () -> ()
    }, {
      %7 = arith.subi %1, %c1 : index
      %8 = arith.divui %7, %c512 : index
      %9 = arith.addi %8, %c1 : index
      %10 = "polygeist.gpu_error"() ({
        %11 = arith.cmpi sge, %0, %c1 : index
        %12 = arith.cmpi sge, %9, %c1 : index
        %13 = arith.andi %11, %12 : i1
        scf.if %13 {
          gpu.launch_func  @__polygeist_gpu_module::@_Z12launch_conv3PfS_S_iii_kernel94180483994384 blocks in (%0, %9, %c1) threads in (%c512, %c1, %c1)  args(%1 : index, %arg5 : i32, %arg3 : i32, %6 : i1, %arg0 : memref<?xf32>, %arg1 : memref<?xf32>, %3 : index, %5 : index, %arg2 : memref<?xf32>)
        }
        "polygeist.polygeist_yield"() : () -> ()
      }) : () -> index
      "polygeist.polygeist_yield"() : () -> ()
    }, {
      %7 = arith.subi %1, %c1 : index
      %8 = arith.divui %7, %c1024 : index
      %9 = arith.addi %8, %c1 : index
      %10 = "polygeist.gpu_error"() ({
        %11 = arith.cmpi sge, %0, %c1 : index
        %12 = arith.cmpi sge, %9, %c1 : index
        %13 = arith.andi %11, %12 : i1
        scf.if %13 {
          gpu.launch_func  @__polygeist_gpu_module::@_Z12launch_conv3PfS_S_iii_kernel94180484011088 blocks in (%0, %9, %c1) threads in (%c1024, %c1, %c1)  args(%1 : index, %arg5 : i32, %arg3 : i32, %6 : i1, %arg0 : memref<?xf32>, %arg1 : memref<?xf32>, %3 : index, %5 : index, %arg2 : memref<?xf32>)
        }
        "polygeist.polygeist_yield"() : () -> ()
      }) : () -> index
      "polygeist.polygeist_yield"() : () -> ()
    }) {alternatives.descs = ["block_size=32,blockDims=x:32;y:1;z:1;,floatOps=32:576;,intOps=4:64;8:320;,loads=4/x:unk|y:unk|z:unk|/0:96;4/x:0|y:0|z:0|/0:480;,stores=4/x:unk|y:unk|z:unk|/0:32;,", "block_size=64,blockDims=x:64;y:1;z:1;,floatOps=32:1152;,intOps=4:128;8:640;,loads=4/x:unk|y:unk|z:unk|/0:192;4/x:0|y:0|z:0|/0:960;,stores=4/x:unk|y:unk|z:unk|/0:64;,", "block_size=128,blockDims=x:128;y:1;z:1;,floatOps=32:2304;,intOps=4:256;8:1280;,loads=4/x:unk|y:unk|z:unk|/0:384;4/x:0|y:0|z:0|/0:1920;,stores=4/x:unk|y:unk|z:unk|/0:128;,", "block_size=256,blockDims=x:256;y:1;z:1;,floatOps=32:4608;,intOps=4:512;8:2560;,loads=4/x:unk|y:unk|z:unk|/0:768;4/x:0|y:0|z:0|/0:3840;,stores=4/x:unk|y:unk|z:unk|/0:256;,", "block_size=512,blockDims=x:512;y:1;z:1;,floatOps=32:9216;,intOps=4:1024;8:5120;,loads=4/x:unk|y:unk|z:unk|/0:1536;4/x:0|y:0|z:0|/0:7680;,stores=4/x:unk|y:unk|z:unk|/0:512;,", "block_size=1024,blockDims=x:1024;y:1;z:1;,floatOps=32:18432;,intOps=4:2048;8:10240;,loads=4/x:unk|y:unk|z:unk|/0:3072;4/x:0|y:0|z:0|/0:15360;,stores=4/x:unk|y:unk|z:unk|/0:1024;,"], alternatives.type = "gpu_kernel", polygeist.altop.id = "+home+yaakov+vortex_hiploc(\22+tmp+polygeist_temp_conv3_kernel.cu\22:38:3)_Z12launch_conv3PfS_S_iii.func.0"} : () -> ()
    return
  }
}
