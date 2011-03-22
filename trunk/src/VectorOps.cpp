#include <gpmr/VectorOps.h>
#include <algorithm>

__host__ int3 imin(const int3 & i0, const int3 & i1) { return int3_ctor(std::min(i0.x, i1.x), std::min(i0.y, i1.y), std::min(i0.z, i1.z)); }
__host__ int3 imax(const int3 & i0, const int3 & i1) { return int3_ctor(std::max(i0.x, i1.x), std::max(i0.y, i1.y), std::max(i0.z, i1.z)); }

__host__ float3 fminf(const float3 & i0, const float3 & i1) { return float3_ctor(std::min(i0.x, i1.x), std::min(i0.y, i1.y), std::min(i0.z, i1.z)); }
__host__ float3 fmaxf(const float3 & i0, const float3 & i1) { return float3_ctor(std::max(i0.x, i1.x), std::max(i0.y, i1.y), std::max(i0.z, i1.z)); }
