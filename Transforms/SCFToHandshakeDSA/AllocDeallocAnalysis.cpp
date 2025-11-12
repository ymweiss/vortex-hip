//===- AllocDeallocAnalysis.cpp - Allocation analysis ----------*- C++ -*-===//
//
// Implementation of allocation/deallocation analysis
//
//===----------------------------------------------------------------------===//

#include "AllocDeallocAnalysis.h"

namespace mlir {
namespace dsa {

void AllocDeallocTracker::analyzeAllocDealloc(ModuleOp module, OpBuilder &builder) {
  module.walk([&](memref::AllocOp allocOp) {
    Value memref = allocOp.getResult();

    // Check if this is a stack allocation marked for deallocation
    if (allocOp->hasAttr("dsa.needs_dealloc")) {
      // Store the memref type (not the operation pointer, which will be invalidated)
      stackAllocTypesNeedingDealloc.push_back(memref.getType());

      // Track memory accesses for this allocation
      bool hasAccess = false;
      for (auto user : memref.getUsers()) {
        if (isa<memref::LoadOp, memref::StoreOp>(user)) {
          hasAccess = true;
          break;
        }
      }
      memrefHasAccess[memref] = hasAccess;

      // Remove the marker attribute
      allocOp->removeAttr("dsa.needs_dealloc");

      // Don't process this allocation in the heap dealloc logic below
      return;
    }

    // Find corresponding dealloc (for heap allocations)
    memref::DeallocOp deallocOp = nullptr;
    Operation *scope = nullptr;

    for (auto user : memref.getUsers()) {
      if (auto dealloc = dyn_cast<memref::DeallocOp>(user)) {
        deallocOp = dealloc;

        // Find the scope (nearest enclosing control flow operation)
        Operation *parent = dealloc->getParentOp();
        while (parent && !isa<scf::ForOp, scf::WhileOp, func::FuncOp>(parent)) {
          parent = parent->getParentOp();
        }
        scope = parent;
        break;
      }
    }

    if (deallocOp) {
      allocToDeallocMap[memref] = deallocOp;
      deallocToScopeMap[deallocOp] = scope;

      // Check if this memref has any load/store accesses
      bool hasAccess = false;
      for (auto user : memref.getUsers()) {
        if (isa<memref::LoadOp, memref::StoreOp>(user)) {
          hasAccess = true;
          break;
        }
      }
      memrefHasAccess[memref] = hasAccess;
    }
  });
}

void AllocDeallocTracker::clear() {
  allocToDeallocMap.clear();
  deallocToScopeMap.clear();
  memrefHasAccess.clear();
  stackAllocTypesNeedingDealloc.clear();
}

} // namespace dsa
} // namespace mlir
