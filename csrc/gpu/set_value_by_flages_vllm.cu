// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/extension.h"

__global__ void set_value_by_flag_and_id_v2(const bool *stop_flags, 
                                            int64_t *pre_ids_all, 
                                            const int64_t *next_tokens, 
                                            const int *seq_lens_encoder, 
                                            const int *seq_lens_decoder, 
                                            const int64_t *step_idx, 
                                            int bs, 
                                            int length) {
    int tid = threadIdx.x;
    if (tid < bs && !stop_flags[tid]) {
        int64_t *pre_ids_all_now = pre_ids_all + tid * length;
        const int seq_len_dec = seq_lens_decoder[tid];
        const int seq_len_enc = seq_lens_encoder[tid];
        const int step_idx_now = step_idx[tid];
        if (seq_len_dec == 0 && seq_len_enc == 0) return; // stoped
        if (step_idx_now >= 0) {
            pre_ids_all_now[step_idx_now] = next_tokens[tid];
        }
    }
}

void SetValueByFlagsAndIdxVLLM(const paddle::Tensor& pre_ids_all, 
                             const paddle::Tensor& next_tokens,
                             const paddle::Tensor& seq_lens_this_time,
                             const paddle::Tensor& seq_lens_encoder,
                             const paddle::Tensor& seq_lens_decoder,
                             const paddle::Tensor& step_idx, 
                             const paddle::Tensor& stop_flags) {
    auto cu_stream = stop_flags.stream();
    std::vector<int64_t> pre_ids_all_shape = pre_ids_all.shape();
    
    int bs = seq_lens_this_time.shape()[0];
    int length = pre_ids_all_shape[1];
    int block_size = (bs + 32 - 1) / 32 * 32;
    set_value_by_flag_and_id_v2<<<1, block_size, 0, cu_stream>>>(stop_flags.data<bool>(), 
                                                                 const_cast<int64_t*>(pre_ids_all.data<int64_t>()), 
                                                                 next_tokens.data<int64_t>(), 
                                                                 seq_lens_encoder.data<int>(),
                                                                 seq_lens_decoder.data<int>(),
                                                                 step_idx.data<int64_t>(), 
                                                                 bs, 
                                                                 length);
}

PD_BUILD_OP(set_value_by_flags_and_idx_vllm)
    .Inputs({"pre_ids_all", "next_tokens", "seq_lens_this_time", "seq_lens_encoder", "seq_lens_decoder", "step_idx", "stop_flags"})
    .Outputs({"pre_ids_all_out"})
    .SetInplaceMap({{"pre_ids_all", "pre_ids_all_out"}})
    .SetKernelFn(PD_KERNEL(SetValueByFlagsAndIdxVLLM));