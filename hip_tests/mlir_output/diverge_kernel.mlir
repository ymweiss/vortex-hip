module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, gpu.container_module, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", polygeist.gpu_module.llvm.data_layout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64", polygeist.gpu_module.llvm.target_triple = "nvptx64-nvidia-cuda", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  gpu.module @__polygeist_gpu_module {
    gpu.func @_Z14launch_divergePiS_jjii_kernel93862240521200(%arg0: index, %arg1: i32, %arg2: i32, %arg3: memref<?xi32>, %arg4: i32, %arg5: memref<?xi32>) kernel attributes {gpu.known_block_size = array<i32: 32, 1, 1>, nvvm.maxntidx = 32 : index, rocdl.max_flat_work_group_size = 32 : index} {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c7_i32 = arith.constant 7 : i32
      %c3_i32 = arith.constant 3 : i32
      %c4_i32 = arith.constant 4 : i32
      %c6_i32 = arith.constant 6 : i32
      %c2_i32 = arith.constant 2 : i32
      %c2147483647_i32 = arith.constant 2147483647 : i32
      %c-1_i32 = arith.constant -1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32
      %c5_i32 = arith.constant 5 : i32
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
      %11 = arith.cmpi slt, %9, %c5_i32 : i32
      %12 = arith.cmpi slt, %9, %arg2 : i32
      %13 = arith.andi %5, %12 : i1
      scf.if %13 {
        %14 = memref.load %arg3[%10] : memref<?xi32>
        %15 = arith.andi %9, %c1_i32 : i32
        %16 = arith.cmpi eq, %15, %c0_i32 : i32
        %17:2 = scf.while (%arg6 = %arg4, %arg7 = %14) : (i32, i32) -> (i32, i32) {
          %35 = arith.cmpi ne, %arg6, %c0_i32 : i32
          %36 = arith.andi %35, %16 : i1
          %37 = scf.if %36 -> (i32) {
            %38 = arith.addi %arg7, %c1_i32 : i32
            scf.yield %38 : i32
          } else {
            scf.yield %arg7 : i32
          }
          scf.condition(%35) %37, %arg6 : i32, i32
        } do {
        ^bb0(%arg6: i32, %arg7: i32):
          %35 = arith.addi %arg7, %c-1_i32 : i32
          scf.yield %35, %arg6 : i32, i32
        }
        %18 = arith.cmpi sge, %9, %c2147483647_i32 : i32
        %19 = scf.if %18 -> (i32) {
          scf.yield %c0_i32 : i32
        } else {
          %35 = arith.addi %17#0, %c2_i32 : i32
          scf.yield %35 : i32
        }
        %20 = arith.cmpi sgt, %9, %c1_i32 : i32
        %21 = scf.if %20 -> (i32) {
          %35 = arith.cmpi sgt, %9, %c2_i32 : i32
          %36 = scf.if %35 -> (i32) {
            %37 = arith.addi %19, %c6_i32 : i32
            scf.yield %37 : i32
          } else {
            %37 = arith.addi %19, %c5_i32 : i32
            scf.yield %37 : i32
          }
          scf.yield %36 : i32
        } else {
          %35 = arith.cmpi sgt, %9, %c0_i32 : i32
          %36 = scf.if %35 -> (i32) {
            %37 = arith.addi %19, %c4_i32 : i32
            scf.yield %37 : i32
          } else {
            %37 = arith.addi %19, %c3_i32 : i32
            scf.yield %37 : i32
          }
          scf.yield %36 : i32
        }
        %22 = arith.cmpi sge, %9, %c0_i32 : i32
        %23 = scf.if %22 -> (i32) {
          %35 = arith.addi %21, %c7_i32 : i32
          scf.yield %35 : i32
        } else {
          scf.yield %c0_i32 : i32
        }
        %24 = scf.for %arg6 = %c0 to %10 step %c1 iter_args(%arg7 = %23) -> (i32) {
          %35 = memref.load %arg3[%arg6] : memref<?xi32>
          %36 = arith.addi %arg7, %35 : i32
          scf.yield %36 : i32
        }
        %25 = scf.execute_region -> i32 {
          cf.switch %9 : i32, [
            default: ^bb4(%24 : i32),
            0: ^bb1,
            1: ^bb2(%24 : i32),
            2: ^bb3(%24, %c3_i32 : i32, i32),
            3: ^bb3(%24, %c5_i32 : i32, i32)
          ]
        ^bb1:  // pred: ^bb0
          %35 = arith.addi %24, %c1_i32 : i32
          cf.br ^bb4(%35 : i32)
        ^bb2(%36: i32):  // pred: ^bb0
          %37 = arith.addi %36, %c-1_i32 : i32
          cf.br ^bb4(%37 : i32)
        ^bb3(%38: i32, %39: i32):  // 2 preds: ^bb0, ^bb0
          %40 = arith.muli %38, %39 : i32
          cf.br ^bb4(%40 : i32)
        ^bb4(%41: i32):  // 4 preds: ^bb0, ^bb1, ^bb2, ^bb3
          scf.yield %41 : i32
        }
        %26 = scf.if %22 -> (i32) {
          %35 = arith.cmpi sgt, %9, %c5_i32 : i32
          %36 = scf.if %35 -> (i32) {
            %37 = memref.load %arg3[%c0] : memref<?xi32>
            scf.yield %37 : i32
          } else {
            scf.yield %9 : i32
          }
          scf.yield %36 : i32
        } else {
          %35 = scf.if %11 -> (i32) {
            %36 = memref.load %arg3[%c1] : memref<?xi32>
            scf.yield %36 : i32
          } else {
            %36 = arith.subi %c0_i32, %9 : i32
            scf.yield %36 : i32
          }
          scf.yield %35 : i32
        }
        %27 = arith.addi %25, %26 : i32
        %28 = memref.load %arg3[%10] : memref<?xi32>
        %29 = arith.cmpi slt, %28, %27 : i32
        %30 = arith.select %29, %28, %27 : i32
        %31 = arith.addi %27, %30 : i32
        %32 = arith.cmpi sgt, %28, %31 : i32
        %33 = arith.select %32, %28, %31 : i32
        %34 = arith.addi %31, %33 : i32
        memref.store %34, %arg5[%10] : memref<?xi32>
      }
      gpu.return
    }
  }
  func.func @_Z14launch_divergePiS_jjii(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c512 = arith.constant 512 : index
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %arg4 : i32 to index
    %1 = arith.index_cast %arg5 : i32 to index
    "polygeist.alternatives"() ({
      %2 = arith.subi %1, %c1 : index
      %3 = arith.divui %2, %c32 : index
      %4 = arith.addi %3, %c1 : index
      %5 = "polygeist.gpu_error"() ({
        %6 = arith.cmpi sge, %0, %c1 : index
        %7 = arith.cmpi sge, %4, %c1 : index
        %8 = arith.andi %6, %7 : i1
        scf.if %8 {
          gpu.launch_func  @__polygeist_gpu_module::@_Z14launch_divergePiS_jjii_kernel93862240521200 blocks in (%0, %4, %c1) threads in (%c32, %c1, %c1)  args(%1 : index, %arg5 : i32, %arg2 : i32, %arg0 : memref<?xi32>, %arg3 : i32, %arg1 : memref<?xi32>)
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
          gpu.launch_func  @__polygeist_gpu_module::@_Z14launch_divergePiS_jjii_kernel93862239529344 blocks in (%0, %4, %c1) threads in (%c64, %c1, %c1)  args(%1 : index, %arg5 : i32, %arg2 : i32, %arg0 : memref<?xi32>, %arg3 : i32, %arg1 : memref<?xi32>)
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
          gpu.launch_func  @__polygeist_gpu_module::@_Z14launch_divergePiS_jjii_kernel93862240277664 blocks in (%0, %4, %c1) threads in (%c128, %c1, %c1)  args(%1 : index, %arg5 : i32, %arg2 : i32, %arg0 : memref<?xi32>, %arg3 : i32, %arg1 : memref<?xi32>)
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
          gpu.launch_func  @__polygeist_gpu_module::@_Z14launch_divergePiS_jjii_kernel93862240462944 blocks in (%0, %4, %c1) threads in (%c256, %c1, %c1)  args(%1 : index, %arg5 : i32, %arg2 : i32, %arg0 : memref<?xi32>, %arg3 : i32, %arg1 : memref<?xi32>)
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
          gpu.launch_func  @__polygeist_gpu_module::@_Z14launch_divergePiS_jjii_kernel93862240491104 blocks in (%0, %4, %c1) threads in (%c512, %c1, %c1)  args(%1 : index, %arg5 : i32, %arg2 : i32, %arg0 : memref<?xi32>, %arg3 : i32, %arg1 : memref<?xi32>)
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
          gpu.launch_func  @__polygeist_gpu_module::@_Z14launch_divergePiS_jjii_kernel93862240519040 blocks in (%0, %4, %c1) threads in (%c1024, %c1, %c1)  args(%1 : index, %arg5 : i32, %arg2 : i32, %arg0 : memref<?xi32>, %arg3 : i32, %arg1 : memref<?xi32>)
        }
        "polygeist.polygeist_yield"() : () -> ()
      }) : () -> index
      "polygeist.polygeist_yield"() : () -> ()
    }) {alternatives.descs = ["block_size=32,blockDims=x:32;y:1;z:1;,floatOps=,intOps=4:576;8:64;,loads=4/x:unk|y:unk|z:unk|/0:64;4/x:0|y:0|z:0|/0:96;,stores=4/x:unk|y:unk|z:unk|/0:32;,", "block_size=64,blockDims=x:64;y:1;z:1;,floatOps=,intOps=4:1152;8:128;,loads=4/x:unk|y:unk|z:unk|/0:128;4/x:0|y:0|z:0|/0:192;,stores=4/x:unk|y:unk|z:unk|/0:64;,", "block_size=128,blockDims=x:128;y:1;z:1;,floatOps=,intOps=4:2304;8:256;,loads=4/x:unk|y:unk|z:unk|/0:256;4/x:0|y:0|z:0|/0:384;,stores=4/x:unk|y:unk|z:unk|/0:128;,", "block_size=256,blockDims=x:256;y:1;z:1;,floatOps=,intOps=4:4608;8:512;,loads=4/x:unk|y:unk|z:unk|/0:512;4/x:0|y:0|z:0|/0:768;,stores=4/x:unk|y:unk|z:unk|/0:256;,", "block_size=512,blockDims=x:512;y:1;z:1;,floatOps=,intOps=4:9216;8:1024;,loads=4/x:unk|y:unk|z:unk|/0:1024;4/x:0|y:0|z:0|/0:1536;,stores=4/x:unk|y:unk|z:unk|/0:512;,", "block_size=1024,blockDims=x:1024;y:1;z:1;,floatOps=,intOps=4:18432;8:2048;,loads=4/x:unk|y:unk|z:unk|/0:2048;4/x:0|y:0|z:0|/0:3072;,stores=4/x:unk|y:unk|z:unk|/0:1024;,"], alternatives.type = "gpu_kernel", polygeist.altop.id = "+home+yaakov+vortex_hiploc(\22+tmp+polygeist_temp_diverge_kernel.cu\22:87:3)_Z14launch_divergePiS_jjii.func.0"} : () -> ()
    return
  }
}
