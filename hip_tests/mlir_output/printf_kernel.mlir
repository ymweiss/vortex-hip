module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, gpu.container_module, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", polygeist.gpu_module.llvm.data_layout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64", polygeist.gpu_module.llvm.target_triple = "nvptx64-nvidia-cuda", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  gpu.module @__polygeist_gpu_module {
    gpu.func @_Z13launch_printfPcjii_kernel94614799239248(%arg0: index, %arg1: i32, %arg2: i32, %arg3: memref<?xi8>, %arg4: !llvm.ptr) kernel attributes {gpu.known_block_size = array<i32: 32, 1, 1>, nvvm.maxntidx = 32 : index, rocdl.max_flat_work_group_size = 32 : index} {
      %c65_i32 = arith.constant 65 : i32
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
        %13 = memref.load %arg3[%10] : memref<?xi8>
        %14 = arith.extsi %13 : i8 to i32
        %15 = arith.addi %14, %c65_i32 : i32
        %16 = arith.trunci %15 : i32 to i8
        %17 = arith.extsi %16 : i8 to i32
        %18 = llvm.call @printf(%arg4, %9, %17) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32, i32) -> i32
      }
      gpu.return
    }
    llvm.func @printf(!llvm.ptr, ...) -> i32
  }
  llvm.mlir.global internal constant @str0("task=%d, value=%c\0A\00") {addr_space = 0 : i32}
  llvm.func @printf(!llvm.ptr, ...) -> i32
  func.func @_Z13launch_printfPcjii(%arg0: memref<?xi8>, %arg1: i32, %arg2: i32, %arg3: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c512 = arith.constant 512 : index
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %arg2 : i32 to index
    %1 = arith.index_cast %arg3 : i32 to index
    %2 = llvm.mlir.addressof @str0 : !llvm.ptr
    %3 = llvm.getelementptr %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<19 x i8>
    "polygeist.alternatives"() ({
      %4 = arith.subi %1, %c1 : index
      %5 = arith.divui %4, %c32 : index
      %6 = arith.addi %5, %c1 : index
      %7 = "polygeist.gpu_error"() ({
        %8 = arith.cmpi sge, %0, %c1 : index
        %9 = arith.cmpi sge, %6, %c1 : index
        %10 = arith.andi %8, %9 : i1
        scf.if %10 {
          gpu.launch_func  @__polygeist_gpu_module::@_Z13launch_printfPcjii_kernel94614799239248 blocks in (%0, %6, %c1) threads in (%c32, %c1, %c1)  args(%1 : index, %arg3 : i32, %arg1 : i32, %arg0 : memref<?xi8>, %3 : !llvm.ptr)
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
          gpu.launch_func  @__polygeist_gpu_module::@_Z13launch_printfPcjii_kernel94614799236352 blocks in (%0, %6, %c1) threads in (%c64, %c1, %c1)  args(%1 : index, %arg3 : i32, %arg1 : i32, %arg0 : memref<?xi8>, %3 : !llvm.ptr)
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
          gpu.launch_func  @__polygeist_gpu_module::@_Z13launch_printfPcjii_kernel94614799244464 blocks in (%0, %6, %c1) threads in (%c128, %c1, %c1)  args(%1 : index, %arg3 : i32, %arg1 : i32, %arg0 : memref<?xi8>, %3 : !llvm.ptr)
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
          gpu.launch_func  @__polygeist_gpu_module::@_Z13launch_printfPcjii_kernel94614799250208 blocks in (%0, %6, %c1) threads in (%c256, %c1, %c1)  args(%1 : index, %arg3 : i32, %arg1 : i32, %arg0 : memref<?xi8>, %3 : !llvm.ptr)
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
          gpu.launch_func  @__polygeist_gpu_module::@_Z13launch_printfPcjii_kernel94614799258208 blocks in (%0, %6, %c1) threads in (%c512, %c1, %c1)  args(%1 : index, %arg3 : i32, %arg1 : i32, %arg0 : memref<?xi8>, %3 : !llvm.ptr)
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
          gpu.launch_func  @__polygeist_gpu_module::@_Z13launch_printfPcjii_kernel94614799266352 blocks in (%0, %6, %c1) threads in (%c1024, %c1, %c1)  args(%1 : index, %arg3 : i32, %arg1 : i32, %arg0 : memref<?xi8>, %3 : !llvm.ptr)
        }
        "polygeist.polygeist_yield"() : () -> ()
      }) : () -> index
      "polygeist.polygeist_yield"() : () -> ()
    }) {alternatives.descs = ["block_size=32,blockDims=x:32;y:1;z:1;,floatOps=,intOps=4:96;8:64;,loads=1/x:unk|y:unk|z:unk|/0:32;,stores=,", "block_size=64,blockDims=x:64;y:1;z:1;,floatOps=,intOps=4:192;8:128;,loads=1/x:unk|y:unk|z:unk|/0:64;,stores=,", "block_size=128,blockDims=x:128;y:1;z:1;,floatOps=,intOps=4:384;8:256;,loads=1/x:unk|y:unk|z:unk|/0:128;,stores=,", "block_size=256,blockDims=x:256;y:1;z:1;,floatOps=,intOps=4:768;8:512;,loads=1/x:unk|y:unk|z:unk|/0:256;,stores=,", "block_size=512,blockDims=x:512;y:1;z:1;,floatOps=,intOps=4:1536;8:1024;,loads=1/x:unk|y:unk|z:unk|/0:512;,stores=,", "block_size=1024,blockDims=x:1024;y:1;z:1;,floatOps=,intOps=4:3072;8:2048;,loads=1/x:unk|y:unk|z:unk|/0:1024;,stores=,"], alternatives.type = "gpu_kernel", polygeist.altop.id = "+home+yaakov+vortex_hiploc(\22+tmp+polygeist_temp_printf_kernel.cu\22:15:3)_Z13launch_printfPcjii.func.0"} : () -> ()
    return
  }
}
