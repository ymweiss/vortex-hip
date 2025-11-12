; ModuleID = 'simple_vortex_style.cpp'
source_filename = "simple_vortex_style.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%union.dim3_t = type { %struct.anon }
%struct.anon = type { i32, i32, i32 }

@blockIdx = external thread_local global %union.dim3_t, align 4
@blockDim = external dso_local global %union.dim3_t, align 4
@threadIdx = external thread_local global %union.dim3_t, align 4

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @_Z6vecaddPfS_S_i(float* %a, float* %b, float* %c, i32 %n) #0 {
entry:
  %a.addr = alloca float*, align 8
  %b.addr = alloca float*, align 8
  %c.addr = alloca float*, align 8
  %n.addr = alloca i32, align 4
  %idx = alloca i32, align 4
  store float* %a, float** %a.addr, align 8
  store float* %b, float** %b.addr, align 8
  store float* %c, float** %c.addr, align 8
  store i32 %n, i32* %n.addr, align 4
  %0 = load i32, i32* getelementptr inbounds (%union.dim3_t, %union.dim3_t* @blockIdx, i32 0, i32 0, i32 0), align 4
  %1 = load i32, i32* getelementptr inbounds (%union.dim3_t, %union.dim3_t* @blockDim, i32 0, i32 0, i32 0), align 4
  %mul = mul i32 %0, %1
  %2 = load i32, i32* getelementptr inbounds (%union.dim3_t, %union.dim3_t* @threadIdx, i32 0, i32 0, i32 0), align 4
  %add = add i32 %mul, %2
  store i32 %add, i32* %idx, align 4
  %3 = load i32, i32* %idx, align 4
  %4 = load i32, i32* %n.addr, align 4
  %cmp = icmp slt i32 %3, %4
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %5 = load float*, float** %a.addr, align 8
  %6 = load i32, i32* %idx, align 4
  %idxprom = sext i32 %6 to i64
  %arrayidx = getelementptr inbounds float, float* %5, i64 %idxprom
  %7 = load float, float* %arrayidx, align 4
  %8 = load float*, float** %b.addr, align 8
  %9 = load i32, i32* %idx, align 4
  %idxprom1 = sext i32 %9 to i64
  %arrayidx2 = getelementptr inbounds float, float* %8, i64 %idxprom1
  %10 = load float, float* %arrayidx2, align 4
  %add3 = fadd float %7, %10
  %11 = load float*, float** %c.addr, align 8
  %12 = load i32, i32* %idx, align 4
  %idxprom4 = sext i32 %12 to i64
  %arrayidx5 = getelementptr inbounds float, float* %11, i64 %idxprom4
  store float %add3, float* %arrayidx5, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @_Z11mixed_typescsilPfPdi(i8 signext %c, i16 signext %s, i32 %i, i64 %l, float* %ptr1, double* %ptr2, i32 %final) #0 {
entry:
  %c.addr = alloca i8, align 1
  %s.addr = alloca i16, align 2
  %i.addr = alloca i32, align 4
  %l.addr = alloca i64, align 8
  %ptr1.addr = alloca float*, align 8
  %ptr2.addr = alloca double*, align 8
  %final.addr = alloca i32, align 4
  store i8 %c, i8* %c.addr, align 1
  store i16 %s, i16* %s.addr, align 2
  store i32 %i, i32* %i.addr, align 4
  store i64 %l, i64* %l.addr, align 8
  store float* %ptr1, float** %ptr1.addr, align 8
  store double* %ptr2, double** %ptr2.addr, align 8
  store i32 %final, i32* %final.addr, align 4
  ret void
}

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 10.0.1 (https://github.com/vortexgpgpu/llvm ef32c611aa214dea855364efd7ba451ec5ec3f74)"}
