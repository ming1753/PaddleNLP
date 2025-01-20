#include "paddle/extension.h"
#include "remote_cache_kv_ipc.h"
#include "paddle/phi/core/allocator.h"
#include "paddle/phi/core/dense_tensor.h"

using cache_write_compelete_signal_type = RemoteCacheKvIpc::save_cache_kv_compelete_signal_layerwise_meta_data;

std::vector<paddle::Tensor> InitSignalLayerwise(const paddle::Tensor& kv_signal_metadata, const int layer_id) {
    auto* kv_signal_metadata_data = reinterpret_cast<const cache_write_compelete_signal_type*>(kv_signal_metadata.data<int8_t>());
    cache_write_compelete_signal_type kv_signal_metadata_tmp = *kv_signal_metadata_data;
    kv_signal_metadata_tmp.layer_id=layer_id;

    int meatedata_size = sizeof(kv_signal_metadata_tmp);
    phi::DenseTensorMeta dmeta(phi::DataType::INT8, phi::make_ddim({meatedata_size}));
    std::shared_ptr<phi::Allocation> alloc(new phi::Allocation((void*)(&kv_signal_metadata_tmp), meatedata_size, phi::CPUPlace()));
    auto kv_signal_metadata_out = paddle::Tensor(std::make_shared<phi::DenseTensor>(alloc, dmeta));
    return {kv_signal_metadata_out};
}

std::vector<std::vector<int64_t>> InitSignalLayerwiseShape(const int ring_id) {
    cache_write_compelete_signal_type kv_signal_metadata;
    int meatedata_size = sizeof(kv_signal_metadata);
    std::vector<int64_t> kv_signal_metadata_shape = {meatedata_size};
    return {kv_signal_metadata_shape};
}

std::vector<paddle::DataType> InitSignalLayerwiseDtype(const int ring_id) {
    return {paddle::DataType::INT8};
}

PD_BUILD_OP(init_signal_layerwise)
    .Inputs({"kv_signal_metadata"})
    .Outputs({"kv_signal_metadata_out"})
    .Attrs({"layer_id: int"})
    .SetKernelFn(PD_KERNEL(InitSignalLayerwise))
    .SetInferShapeFn(PD_INFER_SHAPE(InitSignalLayerwiseShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(InitSignalLayerwiseDtype));