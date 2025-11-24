module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, gpu.container_module, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", polygeist.gpu_module.llvm.data_layout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64", polygeist.gpu_module.llvm.target_triple = "nvptx64-nvidia-cuda", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  gpu.module @__polygeist_gpu_module {
    gpu.func @_Z12launch_sgemvPfS_S_jjii_kernel94204513471632(%arg0: index, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: memref<?xf32>, %arg5: memref<?xf32>, %arg6: index, %arg7: memref<?xf32>) kernel attributes {gpu.known_block_size = array<i32: 32, 1, 1>, nvvm.maxntidx = 32 : index, rocdl.max_flat_work_group_size = 32 : index} {
      %cst = arith.constant 0.000000e+00 : f32
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %c3 = arith.constant 3 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
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
      %11 = arith.cmpi slt, %9, %arg2 : i32
      %12 = arith.andi %5, %11 : i1
      scf.if %12 {
        %13 = arith.muli %9, %arg3 : i32
        %14 = scf.for %arg8 = %c0 to %arg6 step %c4 iter_args(%arg9 = %cst) -> (f32) {
          %15 = arith.index_cast %arg8 : index to i32
          %16 = arith.addi %13, %15 : i32
          %17 = arith.index_cast %16 : i32 to index
          %18 = memref.load %arg4[%17] : memref<?xf32>
          %19 = memref.load %arg5[%arg8] : memref<?xf32>
          %20 = arith.mulf %18, %19 : f32
          %21 = arith.addi %17, %c1 : index
          %22 = memref.load %arg4[%21] : memref<?xf32>
          %23 = arith.addi %arg8, %c1 : index
          %24 = memref.load %arg5[%23] : memref<?xf32>
          %25 = arith.mulf %22, %24 : f32
          %26 = arith.addf %20, %25 : f32
          %27 = arith.addi %17, %c2 : index
          %28 = memref.load %arg4[%27] : memref<?xf32>
          %29 = arith.addi %arg8, %c2 : index
          %30 = memref.load %arg5[%29] : memref<?xf32>
          %31 = arith.mulf %28, %30 : f32
          %32 = arith.addf %26, %31 : f32
          %33 = arith.addi %17, %c3 : index
          %34 = memref.load %arg4[%33] : memref<?xf32>
          %35 = arith.addi %arg8, %c3 : index
          %36 = memref.load %arg5[%35] : memref<?xf32>
          %37 = arith.mulf %34, %36 : f32
          %38 = arith.addf %32, %37 : f32
          %39 = arith.addf %arg9, %38 : f32
          scf.yield %39 : f32
        }
        memref.store %14, %arg7[%10] : memref<?xf32>
      }
      gpu.return
    }
  }
  func.func @_Z12launch_sgemvPfS_S_jjii(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c512 = arith.constant 512 : index
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %arg5 : i32 to index
    %1 = arith.index_cast %arg6 : i32 to index
    %2 = arith.index_cast %arg4 : i32 to index
    "polygeist.alternatives"() ({
      %3 = arith.subi %1, %c1 : index
      %4 = arith.divui %3, %c32 : index
      %5 = arith.addi %4, %c1 : index
      %6 = "polygeist.gpu_error"() ({
        %7 = arith.cmpi sge, %0, %c1 : index
        %8 = arith.cmpi sge, %5, %c1 : index
        %9 = arith.andi %7, %8 : i1
        scf.if %9 {
          gpu.launch_func  @__polygeist_gpu_module::@_Z12launch_sgemvPfS_S_jjii_kernel94204513471632 blocks in (%0, %5, %c1) threads in (%c32, %c1, %c1)  args(%1 : index, %arg6 : i32, %arg3 : i32, %arg4 : i32, %arg0 : memref<?xf32>, %arg1 : memref<?xf32>, %2 : index, %arg2 : memref<?xf32>)
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
          gpu.launch_func  @__polygeist_gpu_module::@_Z12launch_sgemvPfS_S_jjii_kernel94204513482800 blocks in (%0, %5, %c1) threads in (%c64, %c1, %c1)  args(%1 : index, %arg6 : i32, %arg3 : i32, %arg4 : i32, %arg0 : memref<?xf32>, %arg1 : memref<?xf32>, %2 : index, %arg2 : memref<?xf32>)
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
          gpu.launch_func  @__polygeist_gpu_module::@_Z12launch_sgemvPfS_S_jjii_kernel94204513494048 blocks in (%0, %5, %c1) threads in (%c128, %c1, %c1)  args(%1 : index, %arg6 : i32, %arg3 : i32, %arg4 : i32, %arg0 : memref<?xf32>, %arg1 : memref<?xf32>, %2 : index, %arg2 : memref<?xf32>)
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
          gpu.launch_func  @__polygeist_gpu_module::@_Z12launch_sgemvPfS_S_jjii_kernel94204513507808 blocks in (%0, %5, %c1) threads in (%c256, %c1, %c1)  args(%1 : index, %arg6 : i32, %arg3 : i32, %arg4 : i32, %arg0 : memref<?xf32>, %arg1 : memref<?xf32>, %2 : index, %arg2 : memref<?xf32>)
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
          gpu.launch_func  @__polygeist_gpu_module::@_Z12launch_sgemvPfS_S_jjii_kernel94204513522064 blocks in (%0, %5, %c1) threads in (%c512, %c1, %c1)  args(%1 : index, %arg6 : i32, %arg3 : i32, %arg4 : i32, %arg0 : memref<?xf32>, %arg1 : memref<?xf32>, %2 : index, %arg2 : memref<?xf32>)
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
          gpu.launch_func  @__polygeist_gpu_module::@_Z12launch_sgemvPfS_S_jjii_kernel94204513536128 blocks in (%0, %5, %c1) threads in (%c1024, %c1, %c1)  args(%1 : index, %arg6 : i32, %arg3 : i32, %arg4 : i32, %arg0 : memref<?xf32>, %arg1 : memref<?xf32>, %2 : index, %arg2 : memref<?xf32>)
        }
        "polygeist.polygeist_yield"() : () -> ()
      }) : () -> index
      "polygeist.polygeist_yield"() : () -> ()
    }) {alternatives.descs = ["block_size=32,blockDims=x:32;y:1;z:1;,floatOps=32:256;,intOps=4:128;8:256;,loads=4/x:unk|y:unk|z:unk|/0:128;4/x:0|y:0|z:0|/0:128;,stores=4/x:unk|y:unk|z:unk|/0:32;,", "block_size=64,blockDims=x:64;y:1;z:1;,floatOps=32:512;,intOps=4:256;8:512;,loads=4/x:unk|y:unk|z:unk|/0:256;4/x:0|y:0|z:0|/0:256;,stores=4/x:unk|y:unk|z:unk|/0:64;,", "block_size=128,blockDims=x:128;y:1;z:1;,floatOps=32:1024;,intOps=4:512;8:1024;,loads=4/x:unk|y:unk|z:unk|/0:512;4/x:0|y:0|z:0|/0:512;,stores=4/x:unk|y:unk|z:unk|/0:128;,", "block_size=256,blockDims=x:256;y:1;z:1;,floatOps=32:2048;,intOps=4:1024;8:2048;,loads=4/x:unk|y:unk|z:unk|/0:1024;4/x:0|y:0|z:0|/0:1024;,stores=4/x:unk|y:unk|z:unk|/0:256;,", "block_size=512,blockDims=x:512;y:1;z:1;,floatOps=32:4096;,intOps=4:2048;8:4096;,loads=4/x:unk|y:unk|z:unk|/0:2048;4/x:0|y:0|z:0|/0:2048;,stores=4/x:unk|y:unk|z:unk|/0:512;,", "block_size=1024,blockDims=x:1024;y:1;z:1;,floatOps=32:8192;,intOps=4:4096;8:8192;,loads=4/x:unk|y:unk|z:unk|/0:4096;4/x:0|y:0|z:0|/0:4096;,stores=4/x:unk|y:unk|z:unk|/0:1024;,"], alternatives.type = "gpu_kernel", polygeist.altop.id = "+home+yaakov+vortex_hiploc(\22+tmp+polygeist_temp_sgemv_kernel.cu\22:19:3)_Z12launch_sgemvPfS_S_jjii.func.0"} : () -> ()
    return
  }
}
