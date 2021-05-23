"""Schedule for dense operator"""
import logging
from tvm import te, tir
import tvm.autotvm as autotvm
from tvm.autotvm.task.space import SplitEntity
from .. import nn
from .. import tag
from .. import generic
from ..utils import traverse_inline, get_const_tuple

logger = logging.getLogger("topi")

@autotvm.register_topi_compute("dense.cim")
def dense_small_batch(cfg, data, weight, bias=None, out_dtype=None):
    return nn.dense(data, weight, bias, out_dtype)


@autotvm.register_topi_schedule("dense.cim")
def schedule_dense_small_batch(cfg, outs):
    """Schedule float32/64 dense with small batch size"""
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == "dense":
            _schedule_dense_small_batch(cfg, s, op.output(0))

    traverse_inline(s, outs[0].op, _callback)
    return s

def _schedule_dense_small_batch(cfg, s, C):
    A, weights = C.op.input_tensors
    _, in_dim_weights = get_const_tuple(weights.shape)
    _, in_dim_A = get_const_tuple(A.shape)

    if isinstance(in_dim_A, int):
        in_dim = in_dim_A
    elif isinstance(in_dim_weights, int):
        in_dim = in_dim_weights
    else:
        in_dim = None

    if in_dim is not None:
        cfg.define_split("tile_k", in_dim, num_outputs=2)
        if cfg.is_fallback:
            cfg["tile_k"] = SplitEntity([-1, 64] if in_dim > 64 else [1, 64])
        _, kf = cfg["tile_k"].apply(s, C, C.op.reduce_axis[0])
    else:
        tile_k = 64
        _, kf = s[C].split(C.op.reduce_axis[0], tile_k)

    CF = s.rfactor(C, kf)

    if C.op in s.outputs:
        Out = C
    else:
        Out = s.outputs[0].output(0)
        s[C].compute_at(s[Out], s[Out].op.axis[1])
    s[Out].bind(s[Out].op.axis[0], te.thread_axis("blockIdx.y"))
    s[Out].bind(s[Out].op.axis[1], te.thread_axis("blockIdx.x"))

    tx = s[C].op.reduce_axis[0]
    thread_x = te.thread_axis("threadIdx.x")
    s[C].bind(tx, thread_x)
    s[CF].compute_at(s[C], tx)
    s[C].set_store_predicate(thread_x.var.equal(0))
    s[Out].set_store_predicate(thread_x.var.equal(0))