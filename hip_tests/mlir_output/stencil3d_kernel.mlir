module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, gpu.container_module, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", polygeist.gpu_module.llvm.data_layout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64", polygeist.gpu_module.llvm.target_triple = "nvptx64-nvidia-cuda", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  gpu.module @__polygeist_gpu_module {
    gpu.func @_Z16launch_stencil3dPfS_iii_kernel94833514235376(%arg0: index, %arg1: i32, %arg2: i32, %arg3: i1, %arg4: i32, %arg5: memref<?xf32>, %arg6: memref<?xf32>) kernel attributes {gpu.known_block_size = array<i32: 32, 1, 1>, nvvm.maxntidx = 32 : index, rocdl.max_flat_work_group_size = 32 : index} {
      %cst = arith.constant 0.000000e+00 : f32
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c-1 = arith.constant -1 : index
      %c3 = arith.constant 3 : index
      %c0_i32 = arith.constant 0 : i32
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
      %16 = arith.andi %15, %arg3 : i1
      %17 = arith.ori %14, %16 : i1
      %18 = arith.xori %17, %true : i1
      %19 = arith.andi %5, %18 : i1
      scf.if %19 {
        %20:2 = scf.for %arg7 = %c-1 to %c2 step %c1 iter_args(%arg8 = %c0_i32, %arg9 = %cst) -> (i32, f32) {
          %23 = arith.index_cast %arg7 : index to i32
          %24 = arith.cmpi slt, %23, %c0_i32 : i32
          %25 = scf.if %24 -> (i32) {
            scf.yield %c0_i32 : i32
          } else {
            %29 = arith.cmpi sge, %23, %arg2 : i32
            %30 = arith.select %29, %arg4, %23 : i32
            scf.yield %30 : i32
          }
          %26 = arith.muli %25, %arg2 : i32
          %27 = arith.muli %26, %arg2 : i32
          %28:2 = scf.for %arg10 = %c-1 to %c2 step %c1 iter_args(%arg11 = %arg8, %arg12 = %arg9) -> (i32, f32) {
            %29 = arith.index_cast %arg10 : index to i32
            %30 = arith.index_cast %arg11 : i32 to index
            %31 = arith.addi %30, %c3 : index
            %32 = arith.index_cast %31 : index to i32
            %33 = arith.cmpi slt, %29, %c0_i32 : i32
            %34 = scf.if %33 -> (i32) {
              scf.yield %c0_i32 : i32
            } else {
              %38 = arith.cmpi sge, %29, %arg2 : i32
              %39 = arith.select %38, %arg4, %29 : i32
              scf.yield %39 : i32
            }
            %35 = arith.muli %34, %arg2 : i32
            %36 = arith.addi %27, %35 : i32
            %37 = scf.for %arg13 = %c-1 to %c2 step %c1 iter_args(%arg14 = %arg12) -> (f32) {
              %38 = arith.index_cast %arg13 : index to i32
              %39 = arith.addi %9, %38 : i32
              %40 = arith.cmpi slt, %39, %c0_i32 : i32
              %41 = scf.if %40 -> (i32) {
                scf.yield %c0_i32 : i32
              } else {
                %46 = arith.cmpi sge, %39, %arg2 : i32
                %47 = arith.select %46, %arg4, %39 : i32
                scf.yield %47 : i32
              }
              %42 = arith.addi %36, %41 : i32
              %43 = arith.index_cast %42 : i32 to index
              %44 = memref.load %arg5[%43] : memref<?xf32>
              %45 = arith.addf %arg14, %44 : f32
              scf.yield %45 : f32
            }
            scf.yield %32, %37 : i32, f32
          }
          scf.yield %28#0, %28#1 : i32, f32
        }
        %21 = arith.sitofp %20#0 : i32 to f32
        %22 = arith.divf %20#1, %21 : f32
        memref.store %22, %arg6[%10] : memref<?xf32>
      }
      gpu.return
    }
  }
  func.func @_Z16launch_stencil3dPfS_iii(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: i32, %arg3: i32, %arg4: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c512 = arith.constant 512 : index
    %c1024 = arith.constant 1024 : index
    %c0_i32 = arith.constant 0 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %arg3 : i32 to index
    %1 = arith.index_cast %arg4 : i32 to index
    %2 = arith.cmpi sle, %arg2, %c0_i32 : i32
    %3 = arith.addi %arg2, %c-1_i32 : i32
    "polygeist.alternatives"() ({
      %4 = arith.subi %1, %c1 : index
      %5 = arith.divui %4, %c32 : index
      %6 = arith.addi %5, %c1 : index
      %7 = "polygeist.gpu_error"() ({
        %8 = arith.cmpi sge, %0, %c1 : index
        %9 = arith.cmpi sge, %6, %c1 : index
        %10 = arith.andi %8, %9 : i1
        scf.if %10 {
          gpu.launch_func  @__polygeist_gpu_module::@_Z16launch_stencil3dPfS_iii_kernel94833514235376 blocks in (%0, %6, %c1) threads in (%c32, %c1, %c1)  args(%1 : index, %arg4 : i32, %arg2 : i32, %2 : i1, %3 : i32, %arg0 : memref<?xf32>, %arg1 : memref<?xf32>)
        }
        "polygeist.polygeist_yield"() : () -> ()
      }) : () -> index
      "polygeist.polygeist_yield"() : () -> ()
    }, {
      %4 = arith.subi %1, %c1 : index
      %5 = arith.divui %4, %c64 : index
      %6 = arith.addi %5, %c1 : index
      %7 = "polygeist.gpu_error"() ({
        %8 = arith.cmpi sge, %0, %c1 : index
        %9 = arith.cmpi sge, %6, %c1 : index
        %10 = arith.andi %8, %9 : i1
        scf.if %10 {
          gpu.launch_func  @__polygeist_gpu_module::@_Z16launch_stencil3dPfS_iii_kernel94833514157312 blocks in (%0, %6, %c1) threads in (%c64, %c1, %c1)  args(%1 : index, %arg4 : i32, %arg2 : i32, %2 : i1, %3 : i32, %arg0 : memref<?xf32>, %arg1 : memref<?xf32>)
        }
        "polygeist.polygeist_yield"() : () -> ()
      }) : () -> index
      "polygeist.polygeist_yield"() : () -> ()
    }, {
      %4 = arith.subi %1, %c1 : index
      %5 = arith.divui %4, %c128 : index
      %6 = arith.addi %5, %c1 : index
      %7 = "polygeist.gpu_error"() ({
        %8 = arith.cmpi sge, %0, %c1 : index
        %9 = arith.cmpi sge, %6, %c1 : index
        %10 = arith.andi %8, %9 : i1
        scf.if %10 {
          gpu.launch_func  @__polygeist_gpu_module::@_Z16launch_stencil3dPfS_iii_kernel94833514174032 blocks in (%0, %6, %c1) threads in (%c128, %c1, %c1)  args(%1 : index, %arg4 : i32, %arg2 : i32, %2 : i1, %3 : i32, %arg0 : memref<?xf32>, %arg1 : memref<?xf32>)
        }
        "polygeist.polygeist_yield"() : () -> ()
      }) : () -> index
      "polygeist.polygeist_yield"() : () -> ()
    }, {
      %4 = arith.subi %1, %c1 : index
      %5 = arith.divui %4, %c256 : index
      %6 = arith.addi %5, %c1 : index
      %7 = "polygeist.gpu_error"() ({
        %8 = arith.cmpi sge, %0, %c1 : index
        %9 = arith.cmpi sge, %6, %c1 : index
        %10 = arith.andi %8, %9 : i1
        scf.if %10 {
          gpu.launch_func  @__polygeist_gpu_module::@_Z16launch_stencil3dPfS_iii_kernel94833514196064 blocks in (%0, %6, %c1) threads in (%c256, %c1, %c1)  args(%1 : index, %arg4 : i32, %arg2 : i32, %2 : i1, %3 : i32, %arg0 : memref<?xf32>, %arg1 : memref<?xf32>)
        }
        "polygeist.polygeist_yield"() : () -> ()
      }) : () -> index
      "polygeist.polygeist_yield"() : () -> ()
    }, {
      %4 = arith.subi %1, %c1 : index
      %5 = arith.divui %4, %c512 : index
      %6 = arith.addi %5, %c1 : index
      %7 = "polygeist.gpu_error"() ({
        %8 = arith.cmpi sge, %0, %c1 : index
        %9 = arith.cmpi sge, %6, %c1 : index
        %10 = arith.andi %8, %9 : i1
        scf.if %10 {
          gpu.launch_func  @__polygeist_gpu_module::@_Z16launch_stencil3dPfS_iii_kernel94833514215952 blocks in (%0, %6, %c1) threads in (%c512, %c1, %c1)  args(%1 : index, %arg4 : i32, %arg2 : i32, %2 : i1, %3 : i32, %arg0 : memref<?xf32>, %arg1 : memref<?xf32>)
        }
        "polygeist.polygeist_yield"() : () -> ()
      }) : () -> index
      "polygeist.polygeist_yield"() : () -> ()
    }, {
      %4 = arith.subi %1, %c1 : index
      %5 = arith.divui %4, %c1024 : index
      %6 = arith.addi %5, %c1 : index
      %7 = "polygeist.gpu_error"() ({
        %8 = arith.cmpi sge, %0, %c1 : index
        %9 = arith.cmpi sge, %6, %c1 : index
        %10 = arith.andi %8, %9 : i1
        scf.if %10 {
          gpu.launch_func  @__polygeist_gpu_module::@_Z16launch_stencil3dPfS_iii_kernel94833514233936 blocks in (%0, %6, %c1) threads in (%c1024, %c1, %c1)  args(%1 : index, %arg4 : i32, %arg2 : i32, %2 : i1, %3 : i32, %arg0 : memref<?xf32>, %arg1 : memref<?xf32>)
        }
        "polygeist.polygeist_yield"() : () -> ()
      }) : () -> index
      "polygeist.polygeist_yield"() : () -> ()
    }) {alternatives.descs = ["block_size=32,blockDims=x:32;y:1;z:1;,floatOps=32:896;,intOps=4:2560;8:352;,loads=4/x:unk|y:unk|z:unk|/0:864;,stores=4/x:unk|y:unk|z:unk|/0:32;,", "block_size=64,blockDims=x:64;y:1;z:1;,floatOps=32:1792;,intOps=4:5120;8:704;,loads=4/x:unk|y:unk|z:unk|/0:1728;,stores=4/x:unk|y:unk|z:unk|/0:64;,", "block_size=128,blockDims=x:128;y:1;z:1;,floatOps=32:3584;,intOps=4:10240;8:1408;,loads=4/x:unk|y:unk|z:unk|/0:3456;,stores=4/x:unk|y:unk|z:unk|/0:128;,", "block_size=256,blockDims=x:256;y:1;z:1;,floatOps=32:7168;,intOps=4:20480;8:2816;,loads=4/x:unk|y:unk|z:unk|/0:6912;,stores=4/x:unk|y:unk|z:unk|/0:256;,", "block_size=512,blockDims=x:512;y:1;z:1;,floatOps=32:14336;,intOps=4:40960;8:5632;,loads=4/x:unk|y:unk|z:unk|/0:13824;,stores=4/x:unk|y:unk|z:unk|/0:512;,", "block_size=1024,blockDims=x:1024;y:1;z:1;,floatOps=32:28672;,intOps=4:81920;8:11264;,loads=4/x:unk|y:unk|z:unk|/0:27648;,stores=4/x:unk|y:unk|z:unk|/0:1024;,"], alternatives.type = "gpu_kernel", polygeist.altop.id = "+home+yaakov+vortex_hiploc(\22+tmp+polygeist_temp_stencil3d_kernel.cu\22:44:3)_Z16launch_stencil3dPfS_iii.func.0"} : () -> ()
    return
  }
}
