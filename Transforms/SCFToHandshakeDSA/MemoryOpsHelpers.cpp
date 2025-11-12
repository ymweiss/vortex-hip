//===- MemoryOpsHelpers.cpp - Memory operation helpers ----------*- C++ -*-===//
//
// Implementation of helper functions for memory operation conversion
//
//===----------------------------------------------------------------------===//

#include "MemoryOpsHelpers.h"

namespace mlir {
namespace dsa {

//===----------------------------------------------------------------------===//
// Basic Memory Operation Helpers
//===----------------------------------------------------------------------===//

bool isMemoryOp(Operation *op) {
  return isa<memref::LoadOp, memref::StoreOp>(op);
}

bool isAllocOp(Operation *op) {
  return isa<memref::AllocOp, memref::AllocaOp>(op);
}

LogicalResult getOpMemRef(Operation *op, Value &out) {
  out = Value();
  if (auto loadOp = dyn_cast<memref::LoadOp>(op))
    out = loadOp.getMemRef();
  else if (auto storeOp = dyn_cast<memref::StoreOp>(op))
    out = storeOp.getMemRef();

  if (out != Value())
    return success();
  return op->emitOpError("Unknown memory operation type");
}

LogicalResult isValidMemrefType(Location loc, MemRefType type) {
  if (type.getShape().size() != 1)
    return emitError(loc) << "memref must be unidimensional";
  return success();
}

SmallVector<Value> getResultsToMemory(Operation *op) {
  if (auto loadOp = dyn_cast<circt::handshake::LoadOp>(op)) {
    SmallVector<Value> results(loadOp.getAddressResults());
    return results;
  } else if (auto storeOp = dyn_cast<circt::handshake::StoreOp>(op)) {
    SmallVector<Value> results(storeOp.getResults());
    return results;
  }
  return {};
}

Value getOriginalMemRef(Value memref) {
  while (auto defOp = memref.getDefiningOp()) {
    if (auto invariantOp = dyn_cast<dsa::InvariantOp>(defOp)) {
      memref = invariantOp.getA();
    } else {
      break;
    }
  }
  return memref;
}

bool hasDynamicDimensions(MemRefType memrefType) {
  for (int64_t dim : memrefType.getShape()) {
    if (ShapedType::isDynamic(dim)) {
      return true;
    }
  }
  return false;
}

void addValueToOperands(Operation *op, Value val) {
  SmallVector<Value> operands(op->getOperands());
  operands.push_back(val);
  op->setOperands(operands);
}

//===----------------------------------------------------------------------===//
// SCF Path Helpers
//===----------------------------------------------------------------------===//

SmallVector<std::string> getScfPath(Operation *op) {
  SmallVector<std::string> path;
  if (auto pathAttr = op->getAttrOfType<ArrayAttr>("dsa.scf_path")) {
    for (auto attr : pathAttr) {
      if (auto strAttr = dyn_cast<StringAttr>(attr)) {
        std::string segment = strAttr.getValue().str();
        // Skip "top.X" prefix
        if (segment.size() < 4 || segment.substr(0, 4) != "top.") {
          path.push_back(segment);
        }
      }
    }
  }
  return path;
}

ScfLevel ScfLevel::parse(const std::string &levelStr) {
  ScfLevel level;
  size_t dot1 = levelStr.find('.');
  size_t dot2 = levelStr.find('.', dot1 + 1);

  if (dot1 == std::string::npos || dot2 == std::string::npos) {
    return level; // invalid
  }

  level.type = levelStr.substr(0, dot1);
  level.id = levelStr.substr(dot1 + 1, dot2 - dot1 - 1);
  level.branch = levelStr.substr(dot2 + 1);
  level.region = level.type + "." + level.id;
  level.fullRegion = levelStr;
  level.valid = true;

  return level;
}

//===----------------------------------------------------------------------===//
// Control Value Finding
//===----------------------------------------------------------------------===//

template <typename FuncOpT>
Value findControlValue(FuncOpT funcOp, ArrayRef<std::string> parentPath,
                       StringRef scfRegion) {
  Value result;

  DSA_DEBUG_STREAM << "[SCF MEM CTRL] Finding control for scf_region=\""
                   << scfRegion << "\"\n";

  funcOp.walk([&](Operation *op) {
    if (auto regionAttr = op->getAttrOfType<StringAttr>("dsa.scf_region")) {
      if (regionAttr.getValue() == scfRegion) {
        auto opPath = getScfPath(op);

        // Check if scf_path matches parent path
        if (opPath.size() == parentPath.size()) {
          bool matches = true;
          for (size_t i = 0; i < opPath.size(); ++i) {
            if (opPath[i] != parentPath[i]) {
              matches = false;
              break;
            }
          }

          if (matches && op->getNumResults() > 0) {
            // For dsa.gate, use cond result (second output, i1 type)
            // For other ops, use first result
            if (auto gateOp = dyn_cast<dsa::GateOp>(op)) {
              result = gateOp.getCond();
              DSA_DEBUG_STREAM << "[SCF MEM CTRL]   Found dsa.gate cond\n";
            } else {
              result = op->getResult(0);
              DSA_DEBUG_STREAM << "[SCF MEM CTRL]   Found control value\n";
            }
            return WalkResult::interrupt();
          }
        }
      }
    }
    return WalkResult::advance();
  });

  if (!result) {
    DSA_DEBUG_STREAM << "[SCF MEM CTRL]   ERROR: Control not found!\n";
  }

  return result;
}

//===----------------------------------------------------------------------===//
// Extmemory Token Helpers
//===----------------------------------------------------------------------===//

Value getExtmemoryDoneToken(Operation *memOp) {
  if (auto loadOp = dyn_cast<circt::handshake::LoadOp>(memOp)) {
    Value dataInput = loadOp.getOperand(1); // data_from_memory

    // Trace back through forks
    while (dataInput) {
      if (auto forkOp = dataInput.getDefiningOp<circt::handshake::ForkOp>()) {
        dataInput = forkOp.getOperand();
        continue;
      }

      // Check if from extmemory
      if (auto extmemOp = dyn_cast<circt::handshake::ExternalMemoryOp>(
              dataInput.getDefiningOp())) {
        auto portIdxAttr = memOp->getAttrOfType<IntegerAttr>("dsa.load_port_idx");
        if (!portIdxAttr) {
          memOp->emitError("Load missing dsa.load_port_idx");
          return Value();
        }
        unsigned ldPortIdx = portIdxAttr.getInt();
        unsigned ldCount = extmemOp.getLdCount();
        unsigned stCount = extmemOp.getStCount();
        // Format: [load_data(0~ldCount-1), store_done(ldCount~ldCount+stCount-1),
        //          load_done(ldCount+stCount~ldCount+stCount+ldCount-1)]
        unsigned doneIdx = ldCount + stCount + ldPortIdx;
        return extmemOp.getResult(doneIdx);
      }
      break;
    }
  } else if (auto storeOp = dyn_cast<circt::handshake::StoreOp>(memOp)) {
    Value addressOutput = storeOp.getDataResult();

    for (auto user : addressOutput.getUsers()) {
      if (auto extmemOp = dyn_cast<circt::handshake::ExternalMemoryOp>(user)) {
        auto portIdxAttr = memOp->getAttrOfType<IntegerAttr>("dsa.store_port_idx");
        if (!portIdxAttr) {
          memOp->emitError("Store missing dsa.store_port_idx");
          return Value();
        }
        unsigned stPortIdx = portIdxAttr.getInt();
        unsigned ldCount = extmemOp.getLdCount();
        // Format: [load_data, store_done(ldCount~ldCount+stCount-1), load_done]
        unsigned doneIdx = ldCount + stPortIdx;
        return extmemOp.getResult(doneIdx);
      }
    }
  }

  return Value();
}

//===----------------------------------------------------------------------===//
// Path Matching Helpers
//===----------------------------------------------------------------------===//

bool pathMatchesPrefix(ArrayRef<std::string> path,
                       ArrayRef<std::string> prefix) {
  if (path.size() < prefix.size()) return false;
  for (size_t i = 0; i < prefix.size(); ++i) {
    if (path[i] != prefix[i]) return false;
  }
  return true;
}

// Template instantiations
template Value findControlValue<func::FuncOp>(
    func::FuncOp funcOp, ArrayRef<std::string> parentPath, StringRef scfRegion);
template Value findControlValue<circt::handshake::FuncOp>(
    circt::handshake::FuncOp funcOp, ArrayRef<std::string> parentPath, StringRef scfRegion);

} // namespace dsa
} // namespace mlir
