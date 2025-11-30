module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, gpu.container_module, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", polygeist.gpu_module.llvm.data_layout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64", polygeist.gpu_module.llvm.target_triple = "nvptx64-nvidia-cuda", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  gpu.module @__polygeist_gpu_module {
    gpu.func @_Z22launch_metadata_kernelPiS_if_kernel94738444747728(%arg0: i32, %arg1: memref<?xi32>, %arg2: f32, %arg3: memref<?xi32>) kernel attributes {gpu.known_block_size = array<i32: 32, 1, 1>, nvvm.maxntidx = 32 : index, rocdl.max_flat_work_group_size = 32 : index} {
      %c256_i32 = arith.constant 256 : i32
      %c256 = arith.constant 256 : index
      %c32 = arith.constant 32 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %c32 : index
      %4 = arith.addi %3, %2 : index
      %5 = arith.cmpi ult, %4, %c256 : index
      %6 = arith.index_cast %0 : index to i32
      %7 = arith.muli %6, %c256_i32 : i32
      %8 = arith.index_cast %4 : index to i32
      %9 = arith.addi %7, %8 : i32
      %10 = arith.index_cast %9 : i32 to index
      %11 = arith.cmpi slt, %9, %arg0 : i32
      %12 = arith.andi %5, %11 : i1
      scf.if %12 {
        %13 = memref.load %arg1[%10] : memref<?xi32>
        %14 = arith.sitofp %13 : i32 to f32
        %15 = arith.mulf %14, %arg2 : f32
        %16 = arith.fptosi %15 : f32 to i32
        memref.store %16, %arg3[%10] : memref<?xi32>
      }
      gpu.return
    }
  }
  func.func @_Z22launch_metadata_kernelPiS_if(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: i32, %arg3: f32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c32 = arith.constant 32 : index
    %c8 = arith.constant 8 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %c255_i32 = arith.constant 255 : i32
    %c256_i32 = arith.constant 256 : i32
    %0 = arith.addi %arg2, %c255_i32 : i32
    %1 = arith.divsi %0, %c256_i32 : i32
    %2 = arith.index_cast %1 : i32 to index
    "polygeist.alternatives"() ({
      %3 = "polygeist.gpu_error"() ({
        %4 = arith.cmpi sge, %2, %c1 : index
        scf.if %4 {
          gpu.launch_func  @__polygeist_gpu_module::@_Z22launch_metadata_kernelPiS_if_kernel94738444747728 blocks in (%2, %c8, %c1) threads in (%c32, %c1, %c1)  args(%arg2 : i32, %arg0 : memref<?xi32>, %arg3 : f32, %arg1 : memref<?xi32>)
        }
        "polygeist.polygeist_yield"() : () -> ()
      }) : () -> index
      "polygeist.polygeist_yield"() : () -> ()
    }, {
      %3 = "polygeist.gpu_error"() ({
        %4 = arith.cmpi sge, %2, %c1 : index
        scf.if %4 {
          gpu.launch_func  @__polygeist_gpu_module::@_Z22launch_metadata_kernelPiS_if_kernel94738445856112 blocks in (%2, %c4, %c1) threads in (%c64, %c1, %c1)  args(%arg2 : i32, %arg0 : memref<?xi32>, %arg3 : f32, %arg1 : memref<?xi32>)
        }
        "polygeist.polygeist_yield"() : () -> ()
      }) : () -> index
      "polygeist.polygeist_yield"() : () -> ()
    }, {
      %3 = "polygeist.gpu_error"() ({
        %4 = arith.cmpi sge, %2, %c1 : index
        scf.if %4 {
          gpu.launch_func  @__polygeist_gpu_module::@_Z22launch_metadata_kernelPiS_if_kernel94738445873360 blocks in (%2, %c2, %c1) threads in (%c128, %c1, %c1)  args(%arg2 : i32, %arg0 : memref<?xi32>, %arg3 : f32, %arg1 : memref<?xi32>)
        }
        "polygeist.polygeist_yield"() : () -> ()
      }) : () -> index
      "polygeist.polygeist_yield"() : () -> ()
    }, {
      %3 = "polygeist.gpu_error"() ({
        %4 = arith.cmpi sge, %2, %c1 : index
        scf.if %4 {
          gpu.launch_func  @__polygeist_gpu_module::@_Z22launch_metadata_kernelPiS_if_kernel94738445880816 blocks in (%2, %c1, %c1) threads in (%c256, %c1, %c1)  args(%arg2 : i32, %arg0 : memref<?xi32>, %arg3 : f32, %arg1 : memref<?xi32>)
        }
        "polygeist.polygeist_yield"() : () -> ()
      }) : () -> index
      "polygeist.polygeist_yield"() : () -> ()
    }, {
      %3 = arith.subi %2, %c1 : index
      %4 = arith.divui %3, %c2 : index
      %5 = arith.addi %4, %c1 : index
      %6 = "polygeist.gpu_error"() ({
        %7 = arith.cmpi sge, %5, %c1 : index
        scf.if %7 {
          gpu.launch_func  @__polygeist_gpu_module::@_Z22launch_metadata_kernelPiS_if_kernel94738445884512 blocks in (%5, %c1, %c1) threads in (%c2, %c256, %c1)  args(%2 : index, %arg2 : i32, %arg0 : memref<?xi32>, %arg3 : f32, %arg1 : memref<?xi32>)
        }
        "polygeist.polygeist_yield"() : () -> ()
      }) : () -> index
      "polygeist.polygeist_yield"() : () -> ()
    }, {
      %3 = arith.subi %2, %c1 : index
      %4 = arith.divui %3, %c4 : index
      %5 = arith.addi %4, %c1 : index
      %6 = "polygeist.gpu_error"() ({
        %7 = arith.cmpi sge, %5, %c1 : index
        scf.if %7 {
          gpu.launch_func  @__polygeist_gpu_module::@_Z22launch_metadata_kernelPiS_if_kernel94738445874128 blocks in (%5, %c1, %c1) threads in (%c4, %c256, %c1)  args(%2 : index, %arg2 : i32, %arg0 : memref<?xi32>, %arg3 : f32, %arg1 : memref<?xi32>)
        }
        "polygeist.polygeist_yield"() : () -> ()
      }) : () -> index
      "polygeist.polygeist_yield"() : () -> ()
    }) {alternatives.descs = ["block_size=32,blockDims=x:32;y:1;z:1;,floatOps=32:32;,intOps=4:64;8:64;,loads=4/x:unk|y:unk|z:unk|/0:32;,stores=4/x:unk|y:unk|z:unk|/0:32;,", "block_size=64,blockDims=x:64;y:1;z:1;,floatOps=32:64;,intOps=4:128;8:128;,loads=4/x:unk|y:unk|z:unk|/0:64;,stores=4/x:unk|y:unk|z:unk|/0:64;,", "block_size=128,blockDims=x:128;y:1;z:1;,floatOps=32:128;,intOps=4:256;8:256;,loads=4/x:unk|y:unk|z:unk|/0:128;,stores=4/x:unk|y:unk|z:unk|/0:128;,", "block_size=256,blockDims=x:256;y:1;z:1;,floatOps=32:256;,intOps=4:512;,loads=4/x:unk|y:unk|z:unk|/0:256;,stores=4/x:unk|y:unk|z:unk|/0:256;,", "block_size=512,blockDims=x:2;y:256;z:1;,floatOps=32:512;,intOps=4:1024;8:1024;,loads=4/x:unk|y:unk|z:unk|/0:512;,stores=4/x:unk|y:unk|z:unk|/0:512;,", "block_size=1024,blockDims=x:4;y:256;z:1;,floatOps=32:1024;,intOps=4:2048;8:2048;,loads=4/x:unk|y:unk|z:unk|/0:1024;,stores=4/x:unk|y:unk|z:unk|/0:1024;,"], alternatives.type = "gpu_kernel", polygeist.altop.id = "+home+yaakov+vortex_hiploc(\22+tmp+polygeist_temp_metadata_test.cu\22:21:5)_Z22launch_metadata_kernelPiS_if.func.0"} : () -> ()
    return
  }
}
