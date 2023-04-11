#include "nccl.h"

ncclDataType_t HackDataType(ncclDataType_t datatype) {
  switch (datatype) {
    case ncclUint8:
      return ncclFp8E4M3;
    case ncclInt8:
      return ncclFp8E5M2;
    default:
      return datatype;
  }
  return datatype;
}
