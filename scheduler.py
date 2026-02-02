import json

BATCH_SIZE = 256
VLEN = 8
PACK_SIZE = 6
ROUNDS = 16

def get_new_state(pack_size):
  return {
    "LOAD_IDX": {
      "offset": 0,
      "load": 2 * (pack_size / 2),
    },
    "LOAD_VAL": {
      "offset": 0,
      "load": 2 * (pack_size / 2),
    },
    "COMPUTE_NODE_VAL_IDX": {
      "offset": 0,
      # "valu": pack_size,
      "alu": 8 * pack_size,
    },
    "LOAD_NODE_VAL": {
      "offset": 0,
      "load": 2 * 4 * pack_size,
    },
    "COMPUTE_HASH_INPUT": {
      "offset": 0,
      "valu": pack_size,
      # "alu": 8 * pack_size,
    },
    "COMPUTE_HASH": {
      "offset": 0,
      "valu": 12 * 6,
    },
    "COMPUTE_NEXT_NODE": {
      "offset": 0,
      "valu": 6 * 3,
    },
    "COMPUTE_OVERFLOW_CONDITION": {
      "offset": 0,
      "valu": pack_size,
      # "alu": 8 * pack_size,
    },
    "WRAP_ON_OVERFLOW": {
      "offset": 0,
      # "flow": pack_size,
      "valu": pack_size,
    },
    "STORE_IDX": {
      "offset": 0,
      "store": 2 * (pack_size / 2),
    },
    "STORE_VAL": {
      "offset": 0,
      "store": 2 * (pack_size / 2),
    }
  }


SINGLE_ROUND_OPERATIONS = [
  "LOAD_IDX",
  "LOAD_VAL",
  "COMPUTE_NODE_VAL_IDX",
  "LOAD_NODE_VAL",
  "COMPUTE_HASH_INPUT",
  "COMPUTE_HASH",
  "COMPUTE_NEXT_NODE",
  "COMPUTE_OVERFLOW_CONDITION",
  "WRAP_ON_OVERFLOW",
  "STORE_IDX",
  "STORE_VAL",
]


THREADS = []

class ThreadState:

  def __init__(self, pack_size=PACK_SIZE):
    self.pack_size = pack_size
    self.current_op_idx = 0
    self.operations = []

    for _ in range(ROUNDS):
      self.operations.extend(SINGLE_ROUND_OPERATIONS)

    new_state = get_new_state(pack_size)

    self.state = []
    for op in self.operations:
      self.state.append(new_state[op].copy())
  
  def current_op(self):
    return self.operations[self.current_op_idx]

  def current_state(self):
    return self.state[self.current_op_idx]

  def next(self):
    self.current_op_idx += 1

  def is_complete(self):
    return self.current_op_idx >= len(self.operations)


def main():
  full_pack_threads = int(BATCH_SIZE / PACK_SIZE / VLEN)
  for _ in range(full_pack_threads):
    THREADS.append(ThreadState())

  threads_remaining = BATCH_SIZE - full_pack_threads * PACK_SIZE * VLEN
  if threads_remaining > 0:
    new_pack_size = int(threads_remaining / VLEN)
    THREADS.append(ThreadState(new_pack_size))

  instructions = []

  while True:
    all_complete = True

    tick_slots = {
      "alu": 12,
      "valu": 6,
      "flow": 1,
      "load": 2,
      "store": 2,
    }

    tick_ops = []
    additional_ops = []

    for thread_id, thread in enumerate(THREADS):
      if thread.is_complete():
        continue

      all_complete = False

      current_op = thread.current_op()
      slots = thread.current_state()
      offset = slots["offset"]
      
      for key, remaining_slots in slots.items():
        if key == "offset":
          continue

        available_slots = tick_slots[key]
        if available_slots > 0 and remaining_slots > 0:
          new_remaining_slots = max(remaining_slots - available_slots, 0)
          consumed_slots = remaining_slots - new_remaining_slots

          tick_slots[key] -= consumed_slots
          slots[key] = new_remaining_slots

          round_idx = thread.current_op_idx // len(SINGLE_ROUND_OPERATIONS)

          tick_ops.append((thread_id, current_op, offset, thread.pack_size, round_idx, consumed_slots))
          slots["offset"] += 1

          # OP is complete, go to the next op.
          if new_remaining_slots == 0:
            # Add debug ops.
            if current_op == "LOAD_IDX":
              additional_ops.append((thread_id, "DEBUG_LOAD_IDX", 0, thread.pack_size, round_idx, 0))
            elif current_op == "LOAD_VAL":
              additional_ops.append((thread_id, "DEBUG_LOAD_VAL", 0, thread.pack_size, round_idx, 0))
            elif current_op == "LOAD_NODE_VAL":
              additional_ops.append((thread_id, "DEBUG_LOAD_NODE_VAL", 0, thread.pack_size, round_idx, 0))
            elif current_op == "COMPUTE_HASH_INPUT":
              additional_ops.append((thread_id, "DEBUG_COMPUTE_HASH_INPUT", 0, thread.pack_size, round_idx, 0))
            elif current_op == "COMPUTE_HASH":
              additional_ops.append((thread_id, "DEBUG_COMPUTE_HASH", 0, thread.pack_size, round_idx, 0))
            elif current_op == "COMPUTE_NEXT_NODE":
              additional_ops.append((thread_id, "DEBUG_COMPUTE_NEXT_NODE", 0, thread.pack_size, round_idx, 0))
            elif current_op == "WRAP_ON_OVERFLOW":
              additional_ops.append((thread_id, "DEBUG_WRAP_ON_OVERFLOW", 0, thread.pack_size, round_idx, 0))

            # Go to next op in thread.
            thread.next()
          
          # This thread has consumed some slots, allow another thread to consume.
          break
    
    print(tick_ops)
    instructions.append(tick_ops)
    if len(additional_ops) > 0:
      print(additional_ops)
      instructions.append(additional_ops)

    if all_complete:
      break

  with open("program_ir.json", "w+") as f:
    json.dump(instructions, f)
  

if __name__ == "__main__":
  main()
