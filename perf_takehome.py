"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)

# How many VALU operations need to be running at once.
# load, store can only have 2 so they need to be duplicated.
# flow can only have 1, so this needs to be repeated 4 times.
PACK_SIZE = 4

class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        self.hash_map_const_map = None

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            if isinstance(slot, tuple):
                instrs.append({engine: [slot]})
            else:
                instrs.append({engine: slot})
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        print(f"{self.scratch_ptr / SCRATCH_SIZE * 100}% ({self.scratch_ptr}/{SCRATCH_SIZE})")
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []
        
        # Pre-compute broadcasted versions of the scratch_const values.
        if self.hash_map_const_map is None:
            self.hash_map_const_map = {}
            for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                if val1 not in self.hash_map_const_map:
                    self.hash_map_const_map[val1] = self.alloc_scratch("hash_map_const_map_val1", length=VLEN)
                    self.add("valu", ("vbroadcast", self.hash_map_const_map[val1], self.scratch_const(val1)))

                # val3 not needed for even hash stages (they use the multiply add trick instead).
                if val3 not in self.hash_map_const_map and hi % 2 != 0:
                    self.hash_map_const_map[val3] = self.alloc_scratch("hash_map_const_map_val3", length=VLEN)
                    self.add("valu", ("vbroadcast", self.hash_map_const_map[val3], self.scratch_const(val3)))

            self.hash_map_const_map["stage0"] = self.alloc_scratch("stage0", length=VLEN)
            self.add("load", ("const", self.hash_map_const_map["stage0"], 1 + (1 << 12)))
            self.add("valu", ("vbroadcast", self.hash_map_const_map["stage0"], self.hash_map_const_map["stage0"]))

            self.hash_map_const_map["stage2"] = self.alloc_scratch("stage2", length=VLEN)
            self.add("load", ("const", self.hash_map_const_map["stage2"], 1 + (1 << 5)))
            self.add("valu", ("vbroadcast", self.hash_map_const_map["stage2"], self.hash_map_const_map["stage2"]))

            self.hash_map_const_map["stage4"] = self.alloc_scratch("stage4", length=VLEN)
            self.add("load", ("const", self.hash_map_const_map["stage4"], 1 + (1 << 3)))
            self.add("valu", ("vbroadcast", self.hash_map_const_map["stage4"], self.hash_map_const_map["stage4"]))

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            const_val_1 = self.hash_map_const_map[val1]

            if hi == 0 or hi == 2 or hi == 4:
                # HASH_STAGES[0]:
                # a = (a + 0x7ED55D16) + (a << 12) = 0x7ED55D16 + a + (a * 2^12) = 0x7ED55D16 + a * (1 + (1 << 12))

                # HASH_STAGES[2]:
                # a = (a + 0x165667B1) + (a << 5) = a + 0x165667B1 + a * 2^5 = 0x165667B1 + a * (1 + 2^5) = 0x165667B1 + a * (1 + (1 << 5))

                # HASH_STAGES[4]:
                # a = (a + 0xFD7046C5) + (a << 3) = 0xFD7046C5 + a * (1 + (1 << 3))
                slots.append(("valu", [("multiply_add", val_hash_addr + j * VLEN, val_hash_addr + j * VLEN, self.hash_map_const_map[f"stage{hi}"], const_val_1) for j in range(PACK_SIZE)]))
            else:
                const_val_3 = self.hash_map_const_map[val3]

                # HASH_STAGES[1]:
                # a = (a ^ 0xC761C23C) ^ (a >> 19) = ??

                # HASH_STAGES[3]:
                # a = (a + 0xD3A2646C) ^ (a << 9) = (a + 0xD3A2646C) ^ a * (1 << 9)

                # HASH_STAGES[5]:
                # a = (a ^ 0xB55A4F09) ^ (a >> 16) = ??
                slots.append(("valu", [(op1, tmp1 + j * VLEN, val_hash_addr + j * VLEN, const_val_1) for j in range(PACK_SIZE)]))
                slots.append(("valu", [(op3, tmp2 + j * VLEN, val_hash_addr + j * VLEN, const_val_3) for j in range(PACK_SIZE)]))
                slots.append(("valu", [(op2, val_hash_addr + j * VLEN, tmp1 + j * VLEN, tmp2 + j * VLEN) for j in range(PACK_SIZE)]))
            
            for k in range(PACK_SIZE):
                slots.append(("debug", ("vcompare", val_hash_addr + k * VLEN, [(round, i * VLEN * PACK_SIZE + k * VLEN + j, "hash_stage", hi) for j in range(VLEN)])))

        return slots


    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Like reference_kernel2 but building actual instructions.
        Scalar implementation using only scalar ALU and load/store.
        """
        tmp1 = self.alloc_scratch("tmp1", VLEN * PACK_SIZE)
        tmp2 = self.alloc_scratch("tmp2", VLEN * PACK_SIZE)

        # Scratch space addresses
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, VLEN)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))
            self.add("valu", ("vbroadcast", self.scratch[v], self.scratch[v]))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))
        # Any debug engine instruction is ignored by the submission simulator
        self.add("debug", ("comment", "Starting loop"))

        body = []  # array of slots

        # Scalar scratch registers
        tmp_idx = self.alloc_scratch("tmp_idx", VLEN * PACK_SIZE)
        tmp_val = self.alloc_scratch("tmp_val", VLEN * PACK_SIZE)
        tmp_node_val = self.alloc_scratch("tmp_node_val", VLEN * PACK_SIZE)
        tmp_addr = self.alloc_scratch("tmp_addr", VLEN * PACK_SIZE)

        tmp_zero_const = self.alloc_scratch("zero_consts", VLEN)
        tmp_one_const = self.alloc_scratch("one_consts", VLEN)
        tmp_two_const = self.alloc_scratch("two_consts", VLEN)

        # Broadcast const values so they can be used in SIMD instructions.
        self.add("valu", ("vbroadcast", tmp_zero_const, zero_const))
        self.add("valu", ("vbroadcast", tmp_one_const, one_const))
        self.add("valu", ("vbroadcast", tmp_two_const, two_const))

        zero_const = tmp_zero_const
        one_const = tmp_one_const
        two_const = tmp_two_const

        index_consts = [self.scratch_const(i * VLEN) for i in range(int(batch_size / VLEN))]

        # Calculates "inp_indices_p + i" of "mem[inp_indices_p + i]" once and can be reused across rounds.
        inp_indices = [self.alloc_scratch() for i in range(int(batch_size / VLEN))]
        for i in range(int(batch_size / VLEN)):
            self.add("alu", ("+", inp_indices[i], self.scratch["inp_indices_p"], index_consts[i]))

        inp_values = [self.alloc_scratch() for i in range(int(batch_size / VLEN))]
        for i in range(int(batch_size / VLEN)):
            self.add("alu", ("+", inp_values[i], self.scratch["inp_values_p"], index_consts[i]))

        for round in range(rounds):
            for i in range(int(batch_size / VLEN / PACK_SIZE)):
                # ======== "idx = mem[inp_indices_p + i]" =======

                # idx = mem[inp_indices_p + i]
                for j in range(int(PACK_SIZE / 2)):
                    slots = [
                        ("vload", tmp_idx + j * 2 * VLEN + k * VLEN, inp_indices[i * PACK_SIZE + j * 2 + k])
                        for k in range(2)
                    ]
                    body.append(("load", slots))

                for j in range(PACK_SIZE):
                    body.append(("debug", ("vcompare", tmp_idx + j * VLEN, [(round, i * VLEN * PACK_SIZE + j * VLEN + k, "idx") for k in range(VLEN)])))

                # # ======== "val = mem[inp_values_p + i]"

                # # val = mem[inp_values_p + i]
                for j in range(int(PACK_SIZE / 2)):
                    slots = [
                        ("vload", tmp_val + j * 2 * VLEN + k * VLEN, inp_values[i * PACK_SIZE + j * 2 + k])
                        for k in range(2)
                    ]
                    body.append(("load", slots))

                for j in range(PACK_SIZE):
                    body.append(("debug", ("vcompare", tmp_val + j * VLEN, [(round, i * VLEN * PACK_SIZE + j * VLEN + k, "val") for k in range(VLEN)])))

                # # ======== "node_val = mem[forest_values_p + idx]" ========

                # # forest_values_p + idx
                body.append(("valu", [
                    ("+", tmp_addr + j * VLEN, self.scratch["forest_values_p"], tmp_idx + j * VLEN)
                    for j in range(PACK_SIZE)
                ]))

                for k in range(PACK_SIZE):
                    for j in range(int(VLEN / 2)):
                        # Two loads can happen concurrently.
                        body.append(("load", [
                            ("load", tmp_node_val + k * VLEN + j * 2, tmp_addr + k * VLEN + j * 2),
                            ("load", tmp_node_val + k * VLEN + j * 2 + 1, tmp_addr + k * VLEN + j * 2 + 1),
                        ]))   

                for k in range(PACK_SIZE):
                    body.append(("debug", ("vcompare", tmp_node_val + k * VLEN, [(round, i * VLEN * PACK_SIZE + k * VLEN + j, "node_val") for j in range(VLEN)])))

                # # ======== "val = myhash(val ^ node_val)" ===========

                # # val ^ node_val
                body.append(("valu", [("^", tmp_val + j * VLEN, tmp_val + j * VLEN, tmp_node_val + j * VLEN) for j in range(PACK_SIZE)]))
                
                # # val = myhash(val ^ node_val)
                body.extend(self.build_hash(tmp_val, tmp1, tmp2, round, i))

                # # ======== "idx = 2*idx + (1 if val % 2 == 0 else 2)" ========

                # # (1 if val % 2 == 0 else 2) --> (val & 0x1) + 1
                body.append(("valu", [("&", tmp1 + j * VLEN, tmp_val + j * VLEN, one_const) for j in range(PACK_SIZE)]))
                body.append(("valu", [("+", tmp1 + j * VLEN, tmp1 + j * VLEN, one_const) for j in range(PACK_SIZE)]))

                # # idx = 2*idx + (1 if val % 2 == 0 else 2)
                body.append(("valu", [("multiply_add", tmp_idx + j * VLEN, two_const, tmp_idx + j * VLEN, tmp1 + j * VLEN) for j in range(PACK_SIZE)]))

                # # ======== "idx = 0 if idx >= n_nodes else idx" ========

                # # idx >= n_nodes
                body.append(("valu", [("<", tmp1 + j * VLEN, tmp_idx + j * VLEN, self.scratch["n_nodes"]) for j in range(PACK_SIZE)]))

                # # "idx = 0 if idx >= n_nodes else idx"
                for j in range(PACK_SIZE):
                    body.append(("flow", ("vselect", tmp_idx + j * VLEN, tmp1 + j * VLEN, tmp_idx + j * VLEN, zero_const)))

                # # ======== "mem[inp_indices_p + i] = idx" ========
                for j in range(int(PACK_SIZE / 2)):
                    body.append(("store", [("vstore", inp_indices[i * PACK_SIZE + j * 2 + k], tmp_idx + j * 2 * VLEN + k * VLEN) for k in range(2)]))

                # # ======== "mem[inp_values_p + i] = val" ========
                for j in range(int(PACK_SIZE / 2)):
                    body.append(("store", [("vstore", inp_values[i * PACK_SIZE + j * 2 + k], tmp_val + j * 2 * VLEN + k * VLEN) for k in range(2)]))

        body_instrs = self.build(body)
        self.instrs.extend(body_instrs)
        # Required to match with the yield in reference_kernel2
        self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
