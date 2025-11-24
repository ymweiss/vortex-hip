module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, gpu.container_module, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", polygeist.gpu_module.llvm.data_layout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64", polygeist.gpu_module.llvm.target_triple = "nvptx64-nvidia-cuda", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  gpu.module @__polygeist_gpu_module {
    gpu.func @_Z14launch_mstressPjPiS0_jii_kernel94841272726896(%arg0: index, %arg1: i32, %arg2: i32, %arg3: memref<?xi32>, %arg4: memref<?xi32>, %arg5: index, %arg6: memref<?xi32>) kernel attributes {gpu.known_block_size = array<i32: 32, 1, 1>, nvvm.maxntidx = 32 : index, rocdl.max_flat_work_group_size = 32 : index} {
      %c1_i32 = arith.constant 1 : i32
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c32 = arith.constant 32 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %c32 : index
      %4 = arith.addi %3, %2 : index
      %5 = arith.cmpi ult, %4, %arg0 : index
      scf.if %5 {
        %6 = arith.index_cast %0 : index to i32
        %7 = arith.muli %6, %arg1 : i32
        %8 = arith.index_cast %4 : index to i32
        %9 = arith.addi %7, %8 : i32
        %10 = arith.muli %9, %arg2 : i32
        %11 = scf.for %arg7 = %c0 to %arg5 step %c1 iter_args(%arg8 = %c1_i32) -> (i32) {
          %13 = arith.index_cast %arg7 : index to i32
          %14 = arith.addi %10, %13 : i32
          %15 = scf.for %arg9 = %c0 to %c8 step %c1 iter_args(%arg10 = %arg8) -> (i32) {
            %16 = arith.index_cast %arg9 : index to i32
            %17 = arith.addi %14, %16 : i32
            %18 = arith.index_cast %17 : i32 to index
            %19 = memref.load %arg3[%18] : memref<?xi32>
            %20 = arith.index_cast %19 : i32 to index
            %21 = memref.load %arg4[%20] : memref<?xi32>
            %22 = arith.muli %arg10, %21 : i32
            scf.yield %22 : i32
          }
          scf.yield %15 : i32
        }
        %12 = arith.index_cast %9 : i32 to index
        memref.store %11, %arg6[%12] : memref<?xi32>
      }
      gpu.return
    }
  }
  func.func @_Z14launch_mstressPjPiS0_jii(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: memref<?xi32>, %arg3: i32, %arg4: i32, %arg5: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c512 = arith.constant 512 : index
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %arg4 : i32 to index
    %1 = arith.index_cast %arg5 : i32 to index
    %2 = arith.index_cast %arg3 : i32 to index
    "polygeist.alternatives"() ({
      %3 = arith.subi %1, %c1 : index
      %4 = arith.divui %3, %c32 : index
      %5 = arith.addi %4, %c1 : index
      %6 = "polygeist.gpu_error"() ({
        %7 = arith.cmpi sge, %0, %c1 : index
        %8 = arith.cmpi sge, %5, %c1 : index
        %9 = arith.andi %7, %8 : i1
        scf.if %9 {
          gpu.launch_func  @__polygeist_gpu_module::@_Z14launch_mstressPjPiS0_jii_kernel94841272726896 blocks in (%0, %5, %c1) threads in (%c32, %c1, %c1)  args(%1 : index, %arg5 : i32, %arg3 : i32, %arg0 : memref<?xi32>, %arg1 : memref<?xi32>, %2 : index, %arg2 : memref<?xi32>)
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
          gpu.launch_func  @__polygeist_gpu_module::@_Z14launch_mstressPjPiS0_jii_kernel94841273601472 blocks in (%0, %5, %c1) threads in (%c64, %c1, %c1)  args(%1 : index, %arg5 : i32, %arg3 : i32, %arg0 : memref<?xi32>, %arg1 : memref<?xi32>, %2 : index, %arg2 : memref<?xi32>)
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
          gpu.launch_func  @__polygeist_gpu_module::@_Z14launch_mstressPjPiS0_jii_kernel94841273610240 blocks in (%0, %5, %c1) threads in (%c128, %c1, %c1)  args(%1 : index, %arg5 : i32, %arg3 : i32, %arg0 : memref<?xi32>, %arg1 : memref<?xi32>, %2 : index, %arg2 : memref<?xi32>)
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
          gpu.launch_func  @__polygeist_gpu_module::@_Z14launch_mstressPjPiS0_jii_kernel94841273617776 blocks in (%0, %5, %c1) threads in (%c256, %c1, %c1)  args(%1 : index, %arg5 : i32, %arg3 : i32, %arg0 : memref<?xi32>, %arg1 : memref<?xi32>, %2 : index, %arg2 : memref<?xi32>)
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
          gpu.launch_func  @__polygeist_gpu_module::@_Z14launch_mstressPjPiS0_jii_kernel94841273629504 blocks in (%0, %5, %c1) threads in (%c512, %c1, %c1)  args(%1 : index, %arg5 : i32, %arg3 : i32, %arg0 : memref<?xi32>, %arg1 : memref<?xi32>, %2 : index, %arg2 : memref<?xi32>)
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
          gpu.launch_func  @__polygeist_gpu_module::@_Z14launch_mstressPjPiS0_jii_kernel94841273641520 blocks in (%0, %5, %c1) threads in (%c1024, %c1, %c1)  args(%1 : index, %arg5 : i32, %arg3 : i32, %arg0 : memref<?xi32>, %arg1 : memref<?xi32>, %2 : index, %arg2 : memref<?xi32>)
        }
        "polygeist.polygeist_yield"() : () -> ()
      }) : () -> index
      "polygeist.polygeist_yield"() : () -> ()
    }) {alternatives.descs = ["block_size=32,blockDims=x:32;y:1;z:1;,floatOps=,intOps=4:640;8:64;,loads=4/x:unk|y:unk|z:unk|/0:512;,stores=4/x:unk|y:unk|z:unk|/0:32;,", "block_size=64,blockDims=x:64;y:1;z:1;,floatOps=,intOps=4:1280;8:128;,loads=4/x:unk|y:unk|z:unk|/0:1024;,stores=4/x:unk|y:unk|z:unk|/0:64;,", "block_size=128,blockDims=x:128;y:1;z:1;,floatOps=,intOps=4:2560;8:256;,loads=4/x:unk|y:unk|z:unk|/0:2048;,stores=4/x:unk|y:unk|z:unk|/0:128;,", "block_size=256,blockDims=x:256;y:1;z:1;,floatOps=,intOps=4:5120;8:512;,loads=4/x:unk|y:unk|z:unk|/0:4096;,stores=4/x:unk|y:unk|z:unk|/0:256;,", "block_size=512,blockDims=x:512;y:1;z:1;,floatOps=,intOps=4:10240;8:1024;,loads=4/x:unk|y:unk|z:unk|/0:8192;,stores=4/x:unk|y:unk|z:unk|/0:512;,", "block_size=1024,blockDims=x:1024;y:1;z:1;,floatOps=,intOps=4:20480;8:2048;,loads=4/x:unk|y:unk|z:unk|/0:16384;,stores=4/x:unk|y:unk|z:unk|/0:1024;,"], alternatives.type = "gpu_kernel", polygeist.altop.id = "+home+yaakov+vortex_hiploc(\22+tmp+polygeist_temp_mstress_kernel.cu\22:31:3)_Z14launch_mstressPjPiS0_jii.func.0"} : () -> ()
    return
  }
}
