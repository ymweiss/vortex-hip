module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, gpu.container_module, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", polygeist.gpu_module.llvm.data_layout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64", polygeist.gpu_module.llvm.target_triple = "nvptx64-nvidia-cuda", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  gpu.module @__polygeist_gpu_module {
    gpu.func @_Z10launch_ctaPiS_ii_kernel94110928361184(%arg0: index, %arg1: i32, %arg2: memref<?xi32>, %arg3: memref<?xi32>) kernel attributes {gpu.known_block_size = array<i32: 32, 1, 1>, nvvm.maxntidx = 32 : index, rocdl.max_flat_work_group_size = 32 : index} {
      %c32 = arith.constant 32 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %c32 : index
      %4 = arith.addi %3, %2 : index
      %5 = arith.cmpi ult, %4, %arg0 : index
      scf.if %5 {
        %6 = arith.index_cast %0 : index to i32
        %7 = arith.index_cast %4 : index to i32
        %8 = arith.muli %6, %arg1 : i32
        %9 = arith.addi %7, %8 : i32
        %10 = arith.index_cast %9 : i32 to index
        %11 = memref.load %arg2[%4] : memref<?xi32>
        %12 = arith.addi %9, %11 : i32
        memref.store %12, %arg3[%10] : memref<?xi32>
      }
      gpu.return
    }
  }
  func.func @_Z10launch_ctaPiS_ii(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: i32, %arg3: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c512 = arith.constant 512 : index
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %arg2 : i32 to index
    %1 = arith.index_cast %arg3 : i32 to index
    "polygeist.alternatives"() ({
      %2 = arith.subi %1, %c1 : index
      %3 = arith.divui %2, %c32 : index
      %4 = arith.addi %3, %c1 : index
      %5 = "polygeist.gpu_error"() ({
        %6 = arith.cmpi sge, %0, %c1 : index
        %7 = arith.cmpi sge, %4, %c1 : index
        %8 = arith.andi %6, %7 : i1
        scf.if %8 {
          gpu.launch_func  @__polygeist_gpu_module::@_Z10launch_ctaPiS_ii_kernel94110928361184 blocks in (%0, %4, %c1) threads in (%c32, %c1, %c1)  args(%1 : index, %arg3 : i32, %arg0 : memref<?xi32>, %arg1 : memref<?xi32>)
        }
        "polygeist.polygeist_yield"() : () -> ()
      }) : () -> index
      "polygeist.polygeist_yield"() : () -> ()
    }, {
      %2 = arith.subi %1, %c1 : index
      %3 = arith.divui %2, %c64 : index
      %4 = arith.addi %3, %c1 : index
      %5 = "polygeist.gpu_error"() ({
        %6 = arith.cmpi sge, %0, %c1 : index
        %7 = arith.cmpi sge, %4, %c1 : index
        %8 = arith.andi %6, %7 : i1
        scf.if %8 {
          gpu.launch_func  @__polygeist_gpu_module::@_Z10launch_ctaPiS_ii_kernel94110928168368 blocks in (%0, %4, %c1) threads in (%c64, %c1, %c1)  args(%1 : index, %arg3 : i32, %arg0 : memref<?xi32>, %arg1 : memref<?xi32>)
        }
        "polygeist.polygeist_yield"() : () -> ()
      }) : () -> index
      "polygeist.polygeist_yield"() : () -> ()
    }, {
      %2 = arith.subi %1, %c1 : index
      %3 = arith.divui %2, %c128 : index
      %4 = arith.addi %3, %c1 : index
      %5 = "polygeist.gpu_error"() ({
        %6 = arith.cmpi sge, %0, %c1 : index
        %7 = arith.cmpi sge, %4, %c1 : index
        %8 = arith.andi %6, %7 : i1
        scf.if %8 {
          gpu.launch_func  @__polygeist_gpu_module::@_Z10launch_ctaPiS_ii_kernel94110928344384 blocks in (%0, %4, %c1) threads in (%c128, %c1, %c1)  args(%1 : index, %arg3 : i32, %arg0 : memref<?xi32>, %arg1 : memref<?xi32>)
        }
        "polygeist.polygeist_yield"() : () -> ()
      }) : () -> index
      "polygeist.polygeist_yield"() : () -> ()
    }, {
      %2 = arith.subi %1, %c1 : index
      %3 = arith.divui %2, %c256 : index
      %4 = arith.addi %3, %c1 : index
      %5 = "polygeist.gpu_error"() ({
        %6 = arith.cmpi sge, %0, %c1 : index
        %7 = arith.cmpi sge, %4, %c1 : index
        %8 = arith.andi %6, %7 : i1
        scf.if %8 {
          gpu.launch_func  @__polygeist_gpu_module::@_Z10launch_ctaPiS_ii_kernel94110928193680 blocks in (%0, %4, %c1) threads in (%c256, %c1, %c1)  args(%1 : index, %arg3 : i32, %arg0 : memref<?xi32>, %arg1 : memref<?xi32>)
        }
        "polygeist.polygeist_yield"() : () -> ()
      }) : () -> index
      "polygeist.polygeist_yield"() : () -> ()
    }, {
      %2 = arith.subi %1, %c1 : index
      %3 = arith.divui %2, %c512 : index
      %4 = arith.addi %3, %c1 : index
      %5 = "polygeist.gpu_error"() ({
        %6 = arith.cmpi sge, %0, %c1 : index
        %7 = arith.cmpi sge, %4, %c1 : index
        %8 = arith.andi %6, %7 : i1
        scf.if %8 {
          gpu.launch_func  @__polygeist_gpu_module::@_Z10launch_ctaPiS_ii_kernel94110928277680 blocks in (%0, %4, %c1) threads in (%c512, %c1, %c1)  args(%1 : index, %arg3 : i32, %arg0 : memref<?xi32>, %arg1 : memref<?xi32>)
        }
        "polygeist.polygeist_yield"() : () -> ()
      }) : () -> index
      "polygeist.polygeist_yield"() : () -> ()
    }, {
      %2 = arith.subi %1, %c1 : index
      %3 = arith.divui %2, %c1024 : index
      %4 = arith.addi %3, %c1 : index
      %5 = "polygeist.gpu_error"() ({
        %6 = arith.cmpi sge, %0, %c1 : index
        %7 = arith.cmpi sge, %4, %c1 : index
        %8 = arith.andi %6, %7 : i1
        scf.if %8 {
          gpu.launch_func  @__polygeist_gpu_module::@_Z10launch_ctaPiS_ii_kernel94110928325600 blocks in (%0, %4, %c1) threads in (%c1024, %c1, %c1)  args(%1 : index, %arg3 : i32, %arg0 : memref<?xi32>, %arg1 : memref<?xi32>)
        }
        "polygeist.polygeist_yield"() : () -> ()
      }) : () -> index
      "polygeist.polygeist_yield"() : () -> ()
    }) {alternatives.descs = ["block_size=32,blockDims=x:32;y:1;z:1;,floatOps=,intOps=4:96;8:64;,loads=4/x:0|y:0|z:0|/0:32;,stores=4/x:unk|y:unk|z:unk|/0:32;,", "block_size=64,blockDims=x:64;y:1;z:1;,floatOps=,intOps=4:192;8:128;,loads=4/x:0|y:0|z:0|/0:64;,stores=4/x:unk|y:unk|z:unk|/0:64;,", "block_size=128,blockDims=x:128;y:1;z:1;,floatOps=,intOps=4:384;8:256;,loads=4/x:0|y:0|z:0|/0:128;,stores=4/x:unk|y:unk|z:unk|/0:128;,", "block_size=256,blockDims=x:256;y:1;z:1;,floatOps=,intOps=4:768;8:512;,loads=4/x:0|y:0|z:0|/0:256;,stores=4/x:unk|y:unk|z:unk|/0:256;,", "block_size=512,blockDims=x:512;y:1;z:1;,floatOps=,intOps=4:1536;8:1024;,loads=4/x:0|y:0|z:0|/0:512;,stores=4/x:unk|y:unk|z:unk|/0:512;,", "block_size=1024,blockDims=x:1024;y:1;z:1;,floatOps=,intOps=4:3072;8:2048;,loads=4/x:0|y:0|z:0|/0:1024;,stores=4/x:unk|y:unk|z:unk|/0:1024;,"], alternatives.type = "gpu_kernel", polygeist.altop.id = "+home+yaakov+vortex_hiploc(\22+tmp+polygeist_temp_cta_kernel.cu\22:13:3)_Z10launch_ctaPiS_ii.func.0"} : () -> ()
    return
  }
}
