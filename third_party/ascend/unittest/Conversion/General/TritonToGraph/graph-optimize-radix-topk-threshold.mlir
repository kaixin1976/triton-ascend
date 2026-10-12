// RUN: triton-opt %s -graph-optimize='compile-on-910-95=true' --verify-each | FileCheck %s
// RUN: triton-opt %s -graph-optimize='compile-on-910-95=false' --verify-each | FileCheck %s --check-prefix=OTHER
//
// Reduced from one byte-selection round of _radix_topk_threshold, BLOCK_R=8192.
// Preserve its u32 key extraction, zero parking and bin-zero correction.
// All four radix rounds use this structure. This is not an end-to-end kernel.
//
// CHECK-LABEL: tt.func @radix_topk_threshold_round(
// CHECK: arith.constant dense<-1> : tensor<8192xi32>
// CHECK: arith.shrui {{.*}} : tensor<8192xi32>
// CHECK-NOT: tt.reduce
// CHECK: tt.histogram
// CHECK-NOT: tt.reduce
// CHECK-NOT: arith.subi
// CHECK: tt.return
// OTHER-LABEL: tt.func @radix_topk_threshold_round(
// OTHER: arith.shrui {{.*}} : tensor<8192xi32>
// OTHER: tt.histogram
// OTHER: tt.reduce
// OTHER: arith.subi
tt.func @radix_topk_threshold_round(%keys: tensor<8192xi64>, %live: tensor<8192xi1>) -> tensor<256xi32> {
  %u32mask = arith.constant dense<4294967295> : tensor<8192xi64>
  %shift = arith.constant dense<24> : tensor<8192xi64>
  %byte_mask = arith.constant dense<255> : tensor<8192xi64>
  %zero = arith.constant dense<0> : tensor<8192xi32>
  %true = arith.constant dense<true> : tensor<8192xi1>
  %zero_bins = arith.constant dense<0> : tensor<256xi32>
  %bounded = arith.andi %keys, %u32mask : tensor<8192xi64>
  %shifted = arith.shrui %bounded, %shift : tensor<8192xi64>
  %byte = arith.andi %shifted, %byte_mask : tensor<8192xi64>
  %indices = arith.trunci %byte : tensor<8192xi64> to tensor<8192xi32>
  %parked = arith.select %live, %indices, %zero : tensor<8192xi1>, tensor<8192xi32>
  %hist = tt.histogram %parked : tensor<8192xi32> -> tensor<256xi32>
  %invalid = arith.xori %live, %true : tensor<8192xi1>
  %counts = arith.extui %invalid : tensor<8192xi1> to tensor<8192xi32>
  %count = "tt.reduce"(%counts) ({
  ^bb0(%a: i32, %b: i32):
    %sum = arith.addi %a, %b : i32
    tt.reduce.return %sum : i32
  }) {axis = 0 : i32} : (tensor<8192xi32>) -> i32
  %splat = tt.splat %count : i32 -> tensor<256xi32>
  %bins = tt.make_range {start = 0 : i32, end = 256 : i32} : tensor<256xi32>
  %iszero = arith.cmpi eq, %bins, %zero_bins : tensor<256xi32>
  %correction = arith.select %iszero, %splat, %zero_bins : tensor<256xi1>, tensor<256xi32>
  %result = arith.subi %hist, %correction : tensor<256xi32>
  tt.return %result : tensor<256xi32>
}

// CHECK-LABEL: tt.func @keep_signed_shift(
// CHECK: arith.shrsi {{.*}} : tensor<8192xi64>
// OTHER-LABEL: tt.func @keep_signed_shift(
// OTHER: arith.shrsi {{.*}} : tensor<8192xi64>
tt.func @keep_signed_shift(%keys: tensor<8192xi64>) -> tensor<8192xi64> {
  %shift = arith.constant dense<24> : tensor<8192xi64>
  %result = arith.shrsi %keys, %shift : tensor<8192xi64>
  tt.return %result : tensor<8192xi64>
}

// Do not widen a narrowed radix key only to truncate it again after the index
// adjustment. This is the shape emitted by TopK Sampling after key packing.
//
// CHECK-LABEL: tt.func @keep_index_adjustment_in_i32(
// CHECK-NOT: arith.extui
// CHECK: %[[KEYS32:.*]] = arith.trunci %{{.*}} : tensor<128xi64> to tensor<128xi32>
// CHECK-NOT: arith.andi
// CHECK: %[[KEY:.*]] = arith.xori %[[KEYS32]], %{{.*}} : tensor<128xi32>
// CHECK-NOT: arith.extui
// CHECK: %[[INDEX:.*]] = arith.subi %[[KEY]], %{{.*}} : tensor<128xi32>
// CHECK-NOT: arith.extui
// CHECK-NOT: tensor<128xi64> to tensor<128xi32>
// CHECK: tt.return %[[INDEX]] : tensor<128xi32>
// OTHER-LABEL: tt.func @keep_index_adjustment_in_i32(
// OTHER-NOT: arith.extui
// OTHER: arith.subi {{.*}} : tensor<128xi32>
// OTHER-NOT: arith.extui
// OTHER: tt.return {{.*}} : tensor<128xi32>
tt.func @keep_index_adjustment_in_i32(%keys: tensor<128xi64>) -> tensor<128xi32> {
  %mask = arith.constant dense<4294967295> : tensor<128xi64>
  %sign = arith.constant dense<2147483648> : tensor<128xi64>
  %base = arith.constant dense<17> : tensor<128xi64>
  %bounded = arith.andi %keys, %mask : tensor<128xi64>
  %key = arith.xori %bounded, %sign : tensor<128xi64>
  %index64 = arith.subi %key, %base : tensor<128xi64>
  %index32 = arith.trunci %index64 : tensor<128xi64> to tensor<128xi32>
  tt.return %index32 : tensor<128xi32>
}

// Match _unpack_sort_indices exactly:
//   (max_u32 - (keys & max_u32)).to(tl.int32)
//
// CHECK-LABEL: tt.func @unpack_sort_indices_in_i32(
// CHECK-NOT: arith.extui
// CHECK: %[[KEYS32:.*]] = arith.trunci %{{.*}} : tensor<128xi64> to tensor<128xi32>
// CHECK-NOT: arith.andi
// CHECK: %[[INDEX:.*]] = arith.subi %{{.*}}, %[[KEYS32]] : tensor<128xi32>
// CHECK-NOT: arith.extui
// CHECK: tt.return %[[INDEX]] : tensor<128xi32>
// OTHER-LABEL: tt.func @unpack_sort_indices_in_i32(
// OTHER-NOT: arith.extui
// OTHER-NOT: arith.andi
// OTHER: arith.subi {{.*}} : tensor<128xi32>
// OTHER-NOT: arith.extui
// OTHER: tt.return {{.*}} : tensor<128xi32>
tt.func @unpack_sort_indices_in_i32(%keys: tensor<128xi64>) -> tensor<128xi32> {
  %max_u32 = arith.constant dense<4294967295> : tensor<128xi64>
  %masked = arith.andi %keys, %max_u32 : tensor<128xi64>
  %index64 = arith.subi %max_u32, %masked : tensor<128xi64>
  %index32 = arith.trunci %index64 : tensor<128xi64> to tensor<128xi32>
  tt.return %index32 : tensor<128xi32>
}

// Keep the wide subtraction when its result has another user. Replacing only
// the truncating user would retain the i64 subtraction and add duplicate i32
// arithmetic.
//
// CHECK-LABEL: tt.func @keep_multi_use_sub_i64(
// CHECK: %[[INDEX64:.*]] = arith.subi {{.*}} : tensor<128xi64>
// CHECK: %[[INDEX32:.*]] = arith.trunci %[[INDEX64]] : tensor<128xi64> to tensor<128xi32>
// CHECK: tt.return %[[INDEX32]], %[[INDEX64]] : tensor<128xi32>, tensor<128xi64>
// OTHER-LABEL: tt.func @keep_multi_use_sub_i64(
// OTHER: %[[INDEX64:.*]] = arith.subi {{.*}} : tensor<128xi64>
// OTHER: %[[INDEX32:.*]] = arith.trunci %[[INDEX64]] : tensor<128xi64> to tensor<128xi32>
// OTHER: tt.return %[[INDEX32]], %[[INDEX64]] : tensor<128xi32>, tensor<128xi64>
tt.func @keep_multi_use_sub_i64(%keys: tensor<128xi64>) -> (tensor<128xi32>, tensor<128xi64>) {
  %base = arith.constant dense<17> : tensor<128xi64>
  %index64 = arith.subi %keys, %base : tensor<128xi64>
  %index32 = arith.trunci %index64 : tensor<128xi64> to tensor<128xi32>
  tt.return %index32, %index64 : tensor<128xi32>, tensor<128xi64>
}

// Preserve overflow promises: narrowing a subtraction carrying nuw/nsw would
// change the width at which poison is defined.
//
// CHECK-LABEL: tt.func @keep_overflow_sub_i64(
// CHECK: %[[INDEX64:.*]] = arith.subi {{.*}} overflow<nuw> : tensor<128xi64>
// CHECK: %[[INDEX32:.*]] = arith.trunci %[[INDEX64]] : tensor<128xi64> to tensor<128xi32>
// CHECK: tt.return %[[INDEX32]] : tensor<128xi32>
// OTHER-LABEL: tt.func @keep_overflow_sub_i64(
// OTHER: %[[INDEX64:.*]] = arith.subi {{.*}} overflow<nuw> : tensor<128xi64>
// OTHER: %[[INDEX32:.*]] = arith.trunci %[[INDEX64]] : tensor<128xi64> to tensor<128xi32>
// OTHER: tt.return %[[INDEX32]] : tensor<128xi32>
tt.func @keep_overflow_sub_i64(%keys: tensor<128xi64>) -> tensor<128xi32> {
  %base = arith.constant dense<17> : tensor<128xi64>
  %index64 = arith.subi %keys, %base overflow<nuw> : tensor<128xi64>
  %index32 = arith.trunci %index64 : tensor<128xi64> to tensor<128xi32>
  tt.return %index32 : tensor<128xi32>
}
