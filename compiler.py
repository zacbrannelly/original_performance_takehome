from dataclasses import dataclass
from problem import HASH_STAGES
import json
import math

PACK_STRIDE = 6
VLEN = 8

PACK_TMP_IDX_LEN = 4 * VLEN

@dataclass
class JITContext:
  inp_indices: list
  tmp_idx: int
  tmp_idx_pack_stride: int

  inp_values: list
  tmp_val: int
  tmp_val_pack_stride: int

  tmp_node_val: int
  tmp_node_val_pack_stride: int

  tmp_addr: int
  tmp_addr_pack_stride: int

  forest_values_p: int

  val_hash_addr: int
  val_hash_addr_pack_stride: int

  hash_map_const_map: dict

  tmp1: int
  tmp1_pack_stride: int

  tmp2: int
  tmp2_pack_stride: int

  zero_const: int
  one_const: int
  two_const: int

  n_nodes_scratch: int

def compile_load_idx(pack_idx, offset, pack_size, round_idx, context):
  tmp_idx = context.tmp_idx + pack_idx * context.tmp_idx_pack_stride
  slots = [
    (
      "vload",
      tmp_idx + offset * 2 * VLEN + k * VLEN,
      context.inp_indices[pack_idx * PACK_STRIDE + offset * 2 + k]
    )
    for k in range(2)
  ]
  return {
    "load": slots,
  }

def compile_debug_load_idx(pack_idx, offset, pack_size, round_idx, context):
  tmp_idx = context.tmp_idx + pack_idx * context.tmp_idx_pack_stride
  return {
    "debug": [
      (
        "vcompare",
        tmp_idx + j * VLEN,
        [(round_idx, pack_idx * VLEN * PACK_STRIDE + j * VLEN + k, "idx") for k in range(VLEN)]
      )
      for j in range(pack_size)
    ],
  }

def compile_load_val(pack_idx, offset, pack_size, round_idx, context):
  tmp_val = context.tmp_val + pack_idx * context.tmp_val_pack_stride
  slots = [
    (
      "vload",
      tmp_val + offset * 2 * VLEN + k * VLEN,
      context.inp_values[pack_idx * PACK_STRIDE + offset * 2 + k]
    )
    for k in range(2)
  ]
  return {
    "load": slots,
  }

def compile_debug_load_val(pack_idx, offset, pack_size, round_idx, context):
  tmp_val = context.tmp_val + pack_idx * context.tmp_val_pack_stride
  return {
    "debug": [
      (
        "vcompare",
        tmp_val + j * VLEN,
        [(round_idx, pack_idx * VLEN * PACK_STRIDE + j * VLEN + k, "val") for k in range(VLEN)]
      )
      for j in range(pack_size)
    ],
  }

def compile_compute_node_val_idx(pack_idx, offset, pack_size, round_idx, context):
  tmp_addr = context.tmp_addr + pack_idx * context.tmp_addr_pack_stride
  tmp_idx = context.tmp_idx + pack_idx * context.tmp_idx_pack_stride
  slots = [
    (
      "+",
      tmp_addr + j * VLEN,
      context.forest_values_p,
      tmp_idx + j * VLEN
    )
    for j in range(pack_size)
  ]
  return {
    "valu": slots,
  }

def compile_load_node_val(pack_idx, offset, pack_size, round_idx, context):
  tmp_node_val = context.tmp_node_val + pack_idx * context.tmp_node_val_pack_stride
  tmp_addr = context.tmp_addr + pack_idx * context.tmp_addr_pack_stride

  pack_pos = math.floor(offset / 4)
  load_pos = offset % 4

  return {
    "load": [
      ("load", tmp_node_val + pack_pos * VLEN + load_pos * 2, tmp_addr + pack_pos * VLEN + load_pos * 2),
      ("load", tmp_node_val + pack_pos * VLEN + load_pos * 2 + 1, tmp_addr + pack_pos * VLEN + load_pos * 2 + 1),
    ]
  }

def compile_debug_load_node_val(pack_idx, offset, pack_size, round_idx, context):
  tmp_node_val = context.tmp_node_val + pack_idx * context.tmp_node_val_pack_stride

  return {
    "debug": [
      ("vcompare", tmp_node_val + k * VLEN, [(round_idx, pack_idx * VLEN * PACK_STRIDE + k * VLEN + j, "node_val") for j in range(VLEN)])
      for k in range(pack_size)
    ]
  }

def compile_compute_hash_input(pack_idx, offset, pack_size, round_idx, context):
  tmp_val = context.tmp_val + pack_idx * context.tmp_val_pack_stride
  tmp_node_val = context.tmp_node_val + pack_idx * context.tmp_node_val_pack_stride

  return {
    "valu": [
      ("^", tmp_val + j * VLEN, tmp_val + j * VLEN, tmp_node_val + j * VLEN)
      for j in range(pack_size)
    ]
  }

def compile_debug_compute_hash_input(pack_idx, offset, pack_size, round_idx, context):
  tmp_val = context.tmp_val + pack_idx * context.tmp_val_pack_stride
  
  return {
    "debug": [
      ("vcompare", tmp_val + k * VLEN, [(round_idx, pack_idx * VLEN * PACK_STRIDE + k * VLEN + j, "hash_input") for j in range(VLEN)])
      for k in range(pack_size)
    ]
  }

def compile_compute_hash(pack_idx, offset, pack_size, round_idx, context):
  slots = []

  val_hash_addr = context.val_hash_addr + pack_idx * context.val_hash_addr_pack_stride

  if offset == 0 or offset == 4 or offset == 8:
    op1, val1, op2, op3, val3 = HASH_STAGES[int(offset / 2)]
    const_val_1 = context.hash_map_const_map[val1]
    stage_const = context.hash_map_const_map[f"stage{int(offset / 2)}"]

    slots = [
      (
        "multiply_add",
        val_hash_addr + j * VLEN,
        val_hash_addr + j * VLEN,
        stage_const,
        const_val_1
      )
      for j in range(pack_size)
    ]
  else:
    if offset < 4:
      group_idx = 0
    elif offset < 8:
      group_idx = 1
    else:
      group_idx = 2

    base = group_idx * 4 + 1
    op_offset = offset - base

    op1, val1, op2, op3, val3 = HASH_STAGES[group_idx * 2 + 1]
  
    const_val_1 = context.hash_map_const_map[val1]
    const_val_3 = context.hash_map_const_map[val3]

    tmp1 = context.tmp1 + pack_idx * context.tmp1_pack_stride
    tmp2 = context.tmp2 + pack_idx * context.tmp2_pack_stride

    if op_offset == 0:
      slots = [
        (op1, tmp1 + j * VLEN, val_hash_addr + j * VLEN, const_val_1) for j in range(pack_size)
      ]
    elif op_offset == 1:
      slots = [
        (op3, tmp2 + j * VLEN, val_hash_addr + j * VLEN, const_val_3) for j in range(pack_size)
      ]
    else:
      slots = [
        (op2, val_hash_addr + j * VLEN, tmp1 + j * VLEN, tmp2 + j * VLEN) for j in range(pack_size)
      ]

  return {
    "valu": slots,
  }

def compile_debug_compute_hash(pack_idx, offset, pack_size, round_idx, context):
  val_hash_addr = context.val_hash_addr + pack_idx * context.val_hash_addr_pack_stride

  return {
    "debug": [
      ("vcompare", val_hash_addr + k * VLEN, [(round_idx, pack_idx * VLEN * PACK_STRIDE + k * VLEN + j, "hash_stage", 5) for j in range(VLEN)])
      for k in range(pack_size)
    ]
  }

def compile_compute_next_node(pack_idx, offset, pack_size, round_idx, context):
  slots = []

  tmp1 = context.tmp1 + pack_idx * context.tmp1_pack_stride
  tmp_val = context.tmp_val + pack_idx * context.tmp_val_pack_stride
  tmp_idx = context.tmp_idx + pack_idx * context.tmp_idx_pack_stride

  if offset == 0:
    slots = [
      ("&", tmp1 + j * VLEN, tmp_val + j * VLEN, context.one_const)
      for j in range(pack_size)
    ]
  elif offset == 1:
    slots = [
      ("+", tmp1 + j * VLEN, tmp1 + j * VLEN, context.one_const)
      for j in range(pack_size)
    ]
  else:
    slots = [
      ("multiply_add", tmp_idx + j * VLEN, context.two_const, tmp_idx + j * VLEN, tmp1 + j * VLEN) 
      for j in range(pack_size)
    ]

  return {
    "valu": slots,
  }

def compile_debug_compute_next_node(pack_idx, offset, pack_size, round_idx, context):
  tmp_idx = context.tmp_idx + pack_idx * context.tmp_idx_pack_stride

  return {
    "debug": [
      ("vcompare", tmp_idx + j * VLEN, [(round_idx, pack_idx * PACK_STRIDE * VLEN + j * VLEN + k, "next_idx") for k in range(VLEN)])
      for j in range(pack_size)
    ]
  }

def compile_compute_overflow_condition(pack_idx, offset, pack_size, round_idx, context):
  tmp1 = context.tmp1 + pack_idx * context.tmp1_pack_stride
  tmp_idx = context.tmp_idx + pack_idx * context.tmp_idx_pack_stride
  return {
    "valu": [
      ("<", tmp1 + j * VLEN, tmp_idx + j * VLEN, context.n_nodes_scratch)
      for j in range(pack_size)
    ]
  }
  
def compile_wrap_on_overflow(pack_idx, offset, pack_size, round_idx, context):
  tmp_idx = context.tmp_idx + pack_idx * context.tmp_idx_pack_stride
  tmp1 = context.tmp1 + pack_idx * context.tmp1_pack_stride

  return {
    "flow": [
      ("vselect", tmp_idx + offset * VLEN, tmp1 + offset * VLEN, tmp_idx + offset * VLEN, context.zero_const)
    ]
  }

def compile_debug_wrap_on_overflow(pack_idx, offset, pack_size, round_idx, context):
  tmp_idx = context.tmp_idx + pack_idx * context.tmp_idx_pack_stride
  return {
    "debug": [
      ("vcompare", tmp_idx + j * VLEN, [(round_idx, pack_idx * PACK_STRIDE * VLEN + j * VLEN + k, "wrapped_idx") for k in range(VLEN)])
      for j in range(pack_size)
    ]
  }

def compile_store_idx(pack_idx, offset, pack_size, round_idx, context):
  tmp_idx = context.tmp_idx + pack_idx * context.tmp_idx_pack_stride
  return {
    "store": [
      ("vstore", context.inp_indices[pack_idx * PACK_STRIDE + offset * 2 + k], tmp_idx + offset * 2 * VLEN + k * VLEN)
      for k in range(2)
    ]
  }

def compile_store_val(pack_idx, offset, pack_size, round_idx, context):
  tmp_val = context.tmp_val + pack_idx * context.tmp_val_pack_stride
  return {
    "store": [
      ("vstore", context.inp_values[pack_idx * PACK_STRIDE + offset * 2 + k], tmp_val + offset * 2 * VLEN + k * VLEN)
      for k in range(2)
    ]
  }

COMPILE_MAP = {
  "LOAD_IDX": compile_load_idx,
  "DEBUG_LOAD_IDX": compile_debug_load_idx,
  "LOAD_VAL": compile_load_val,
  "DEBUG_LOAD_VAL": compile_debug_load_val,
  "COMPUTE_NODE_VAL_IDX": compile_compute_node_val_idx,
  "LOAD_NODE_VAL": compile_load_node_val,
  "DEBUG_LOAD_NODE_VAL": compile_debug_load_node_val,
  "COMPUTE_HASH_INPUT": compile_compute_hash_input,
  "DEBUG_COMPUTE_HASH_INPUT": compile_debug_compute_hash_input,
  "COMPUTE_HASH": compile_compute_hash,
  "DEBUG_COMPUTE_HASH": compile_debug_compute_hash,
  "COMPUTE_NEXT_NODE": compile_compute_next_node,
  "DEBUG_COMPUTE_NEXT_NODE": compile_debug_compute_next_node,
  "COMPUTE_OVERFLOW_CONDITION": compile_compute_overflow_condition,
  "WRAP_ON_OVERFLOW": compile_wrap_on_overflow,
  "DEBUG_WRAP_ON_OVERFLOW": compile_debug_wrap_on_overflow,
  "STORE_IDX": compile_store_idx,
  "STORE_VAL": compile_store_val,
}

def compile_jit_code(context: JITContext):
  with open("program_ir.json", "r") as f:
    program_ir = json.load(f)

  output_code = []
  
  for slots in program_ir:
    instruction = {}
    for pack_idx, operation, offset, pack_size, round_idx in slots:
      if operation not in COMPILE_MAP:
        continue
      compiled_slots = COMPILE_MAP[operation](pack_idx, offset, pack_size, round_idx, context)
      instruction = {**instruction, **compiled_slots}

    output_code.append(instruction)

  with open("program_code.json", "w") as f:
    json.dump(output_code, f, indent=2)

  return output_code

def main():
  context = JITContext(
    inp_indices = [0] * 2000,
    tmp_idx = 0,
    tmp_idx_pack_stride = 0,
    inp_values = [0] * 2000,
    tmp_val = 0,
    tmp_val_pack_stride = 0,
    tmp_node_val=0,
    tmp_node_val_pack_stride=0,
    tmp_addr = 0,
    tmp_addr_pack_stride = 0,
    forest_values_p = 0,
    val_hash_addr = 0,
    val_hash_addr_pack_stride = 0,
    hash_map_const_map = {
      9: 0,
      16: 0,
      19: 0,
      4251993797: 0,
      3042594569: 0,
      2127912214: 0,
      3345072700: 0,
      374761393: 0,
      3550635116: 0,
      "stage0": 0,
      "stage1": 0,
      "stage2": 0,
      "stage3": 0,
      "stage4": 0,
      "stage5": 0,
    },
    tmp1 = 0,
    tmp1_pack_stride = 0,
    tmp2 = 0,
    tmp2_pack_stride = 0,
    zero_const = 0,
    one_const = 0,
    two_const = 0,
    n_nodes_scratch = 0,
  )
  compile_jit_code(context)

if __name__ == "__main__":
  main()
