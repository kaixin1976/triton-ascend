// RUN: triton-opt --split-input-file --verify-diagnostics %s

module {
  tt.func public @bad_count_type(
      %weights: !tt.ptr<f16>, %tokens: !tt.ptr<f16>,
      %counts: !tt.ptr<f16>, %output: !tt.ptr<f16>) {
    // expected-error @+1 {{'ascend.grouped_dot' op tokens_per_expert pointer must have an integer pointee}}
    ascend.grouped_dot %weights, %tokens, %counts, %output {
        num_experts = 4 : i64, max_tokens = 512 : i64,
        out_features = 128 : i64, in_features = 64 : i64
    } : !tt.ptr<f16>, !tt.ptr<f16>, !tt.ptr<f16>, !tt.ptr<f16>
    tt.return
  }
}

// -----

module {
  tt.func public @bad_mx_data_type(
      %weights: !tt.ptr<f16>, %tokens: !tt.ptr<f16>,
      %token_scale: !tt.ptr<i8>, %weight_scale: !tt.ptr<i8>,
      %counts: !tt.ptr<i64>, %output: !tt.ptr<f16>,
      %output_scale: !tt.ptr<i8>) {
    // expected-error @+1 {{'ascend.grouped_dot_swiglu_mx' op weights, tokens and output must have the same FP8 pointee type}}
    ascend.grouped_dot_swiglu_mx
        %weights, %tokens, %token_scale, %weight_scale, %counts, %output,
        %output_scale {
          num_experts = 8 : i64, max_tokens = 384 : i64,
          out_features = 1024 : i64, in_features = 4096 : i64
        } : !tt.ptr<f16>, !tt.ptr<f16>, !tt.ptr<i8>, !tt.ptr<i8>,
            !tt.ptr<i64>, !tt.ptr<f16>, !tt.ptr<i8>
    tt.return
  }
}

// -----

module {
  tt.func public @bad_mx_group_shape(
      %weights: !tt.ptr<f8E4M3FN>, %tokens: !tt.ptr<f8E4M3FN>,
      %token_scale: !tt.ptr<i8>, %weight_scale: !tt.ptr<i8>,
      %counts: !tt.ptr<i64>, %output: !tt.ptr<f8E4M3FN>,
      %output_scale: !tt.ptr<i8>) {
    // expected-error @+1 {{'ascend.grouped_dot_swiglu_mx' op N and K must be divisible by the MX scale group size 32}}
    ascend.grouped_dot_swiglu_mx
        %weights, %tokens, %token_scale, %weight_scale, %counts, %output,
        %output_scale {
          num_experts = 8 : i64, max_tokens = 384 : i64,
          out_features = 1000 : i64, in_features = 4096 : i64
        } : !tt.ptr<f8E4M3FN>, !tt.ptr<f8E4M3FN>, !tt.ptr<i8>,
            !tt.ptr<i8>, !tt.ptr<i64>, !tt.ptr<f8E4M3FN>, !tt.ptr<i8>
    tt.return
  }
}

// -----

module {
  tt.func public @bad_static_shape(
      %weights: !tt.ptr<f16>, %tokens: !tt.ptr<f16>,
      %counts: !tt.ptr<i32>, %output: !tt.ptr<f16>) {
    // expected-error @+1 {{'ascend.grouped_dot' op E, M_max, N, and K must all be positive}}
    ascend.grouped_dot %weights, %tokens, %counts, %output {
        num_experts = 4 : i64, max_tokens = 0 : i64,
        out_features = 128 : i64, in_features = 64 : i64
    } : !tt.ptr<f16>, !tt.ptr<f16>, !tt.ptr<i32>, !tt.ptr<f16>
    tt.return
  }
}
