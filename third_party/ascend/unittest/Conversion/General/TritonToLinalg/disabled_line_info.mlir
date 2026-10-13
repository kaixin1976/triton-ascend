// RUN: env TRITON_DISABLE_LINE_INFO=1 triton-opt --triton-to-linalg \
// RUN:   --mlir-print-debuginfo %s 2>&1 | FileCheck %s

// CHECK-NOT: all have no location
// CHECK-LABEL: func.func @disabled_line_info
// CHECK: arith.addi
// CHECK: return
// CHECK-NOT: all have no location

module {
  tt.func @disabled_line_info(%arg0: i32 loc(unknown),
                              %arg1: i32 loc(unknown)) -> i32 {
    %0 = arith.addi %arg0, %arg1 : i32 loc(unknown)
    tt.return %0 : i32 loc(unknown)
  } loc(unknown)
} loc(unknown)
