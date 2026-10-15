// RUN: triton-opt --triton-to-linalg="named-ops=True" %s | FileCheck %s

module {
  tt.func public @dynamic_m_grouped_dot(
      %weights: !tt.ptr<f16>, %tokens: !tt.ptr<f16>,
      %counts: !tt.ptr<i64>, %output: !tt.ptr<f16>) {
    ascend.grouped_dot %weights, %tokens, %counts, %output {
        num_experts = 4 : i64, max_tokens = 512 : i64,
        out_features = 128 : i64, in_features = 64 : i64
    } : !tt.ptr<f16>, !tt.ptr<f16>, !tt.ptr<i64>, !tt.ptr<f16>
    tt.return
  }
}

// -----

module {
  tt.func public @dynamic_m_grouped_dot_swiglu_mx(
      %weights: !tt.ptr<f8E4M3FN>, %tokens: !tt.ptr<f8E4M3FN>,
      %token_scale: !tt.ptr<i8>, %weight_scale: !tt.ptr<i8>,
      %counts: !tt.ptr<i64>, %output: !tt.ptr<f8E4M3FN>,
      %output_scale: !tt.ptr<i8>) {
    ascend.grouped_dot_swiglu_mx
        %weights, %tokens, %token_scale, %weight_scale, %counts, %output,
        %output_scale {
          num_experts = 8 : i64, max_tokens = 384 : i64,
          out_features = 1024 : i64, in_features = 4096 : i64,
          control_workers = 0 : i64
        } : !tt.ptr<f8E4M3FN>, !tt.ptr<f8E4M3FN>, !tt.ptr<i8>,
            !tt.ptr<i8>, !tt.ptr<i64>, !tt.ptr<f8E4M3FN>, !tt.ptr<i8>
    tt.return
  }
}

// CHECK-LABEL: func.func @dynamic_m_grouped_dot
// CHECK: %[[WEIGHTS:.*]] = memref.reinterpret_cast %{{.*}} to offset: [0], sizes: [4, 128, 64], strides: [8192, 64, 1] : memref<?xf16> to memref<4x128x64xf16>
// CHECK: %[[TOKENS:.*]] = memref.reinterpret_cast %{{.*}} to offset: [0], sizes: [512, 64], strides: [64, 1] : memref<?xf16> to memref<512x64xf16>
// CHECK: %[[COUNTS:.*]] = memref.reinterpret_cast %{{.*}} to offset: [0], sizes: [4], strides: [1] : memref<?xi64> to memref<4xi64>
// CHECK: %[[OUTPUT:.*]] = memref.reinterpret_cast %{{.*}} to offset: [0], sizes: [512, 128], strides: [128, 1] : memref<?xf16> to memref<512x128xf16>
// CHECK: hivm.hir.mix_group_matmul ins(%[[WEIGHTS]], %[[TOKENS]], %[[COUNTS]] : memref<4x128x64xf16>, memref<512x64xf16>, memref<4xi64>) outs(%[[OUTPUT]] : memref<512x128xf16>)
// CHECK: return

// CHECK-LABEL: func.func @dynamic_m_grouped_dot_swiglu_mx
// CHECK: %[[WEIGHTS_MX:.*]] = memref.reinterpret_cast %{{.*}} to offset: [0], sizes: [8, 2048, 4096], strides: [8388608, 4096, 1] : memref<?xf8E4M3FN> to memref<8x2048x4096xf8E4M3FN>
// CHECK: %[[TOKENS_MX:.*]] = memref.reinterpret_cast %{{.*}} to offset: [0], sizes: [384, 4096], strides: [4096, 1] : memref<?xf8E4M3FN> to memref<384x4096xf8E4M3FN>
// CHECK: %[[TOKEN_SCALE:.*]] = memref.reinterpret_cast %{{.*}} to offset: [0], sizes: [384, 128], strides: [128, 1] : memref<?xi8> to memref<384x128xi8>
// CHECK: %[[WEIGHT_SCALE:.*]] = memref.reinterpret_cast %{{.*}} to offset: [0], sizes: [8, 2048, 128], strides: [262144, 128, 1] : memref<?xi8> to memref<8x2048x128xi8>
// CHECK: %[[COUNTS_MX:.*]] = memref.reinterpret_cast %{{.*}} to offset: [0], sizes: [8], strides: [1] : memref<?xi64> to memref<8xi64>
// CHECK: %[[OUTPUT_MX:.*]] = memref.reinterpret_cast %{{.*}} to offset: [0], sizes: [384, 1024], strides: [1024, 1] : memref<?xf8E4M3FN> to memref<384x1024xf8E4M3FN>
// CHECK: %[[OUTPUT_SCALE:.*]] = memref.reinterpret_cast %{{.*}} to offset: [0], sizes: [384, 32], strides: [32, 1] : memref<?xi8> to memref<384x32xi8>
// CHECK: hivm.hir.mix_group_matmul {post_vector_func = "swiglu_mx_quant"} ins(%[[WEIGHTS_MX]], %[[TOKENS_MX]], %[[COUNTS_MX]] : memref<8x2048x4096xf8E4M3FN>, memref<384x4096xf8E4M3FN>, memref<8xi64>) post_vector_func_ins(%[[TOKEN_SCALE]], %[[WEIGHT_SCALE]] : memref<384x128xi8>, memref<8x2048x128xi8>) post_vector_func_outs(%[[OUTPUT_SCALE]] : memref<384x32xi8>) outs(%[[OUTPUT_MX]] : memref<384x1024xf8E4M3FN>)
// CHECK: return
