from tvm import topi
from tvm.auto_scheduler import is_auto_scheduler_enabled
from tvm.contrib import nvcc
from tvm.contrib.thrust import can_use_thrust
from tvm.te import SpecializedCondition

from .. import op as _op
from .generic import *

@conv2d_strategy.register("cim")
def conv2d_strategy_cim(attrs, inputs, out_type, target):
    """conv2d cuda strategy"""
    strategy = _op.OpStrategy()
    data, kernel = inputs
    stride_h, stride_w = attrs.get_int_tuple("strides")
    dilation_h, dilation_w = attrs.get_int_tuple("dilation")
    padding = attrs.get_int_tuple("padding")
    groups = attrs.groups
    layout = attrs.data_layout
    kernel_layout = attrs.kernel_layout
    if dilation_h < 1 or dilation_w < 1:
        raise ValueError("dilation should be positive value")

    if groups == 1:
        if layout == "NCHW":
            assert kernel_layout == "OIHW"
            if data.dtype in ("int8", "uint8") and kernel.dtype in ("int8", "uint8"):
                assert data.dtype == kernel.dtype
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.cuda.conv2d_nchw_int8),
                    wrap_topi_schedule(topi.cuda.schedule_conv2d_nchw_int8),
                    name="conv2d_nchw_int8.cuda",
                )
            else:
                strategy.add_implementation(                            #change
                    wrap_compute_conv2d(topi.cim.conv2d_nchw),
                    wrap_topi_schedule(topi.cim.schedule_conv2d_nchw),
                    name="conv2d_nchw.cim",
                )
            _, _, kh, kw = get_const_tuple(kernel.shape)
            if (
                (2 < kh < 8 and 2 < kw < 8 and kh == kw)
                and (stride_h == 1 and stride_w == 1)
                and (dilation_h == 1 and dilation_w == 1)
            ):
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.cuda.conv2d_nchw_winograd),
                    wrap_topi_schedule(topi.cuda.schedule_conv2d_nchw_winograd),
                    name="conv2d_nchw_winograd.cuda",
                    plevel=5,
                )
        elif layout == "HWCN":
            assert kernel_layout == "HWIO"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.cuda.conv2d_hwcn),
                wrap_topi_schedule(topi.cuda.schedule_conv2d_hwcn),
                name="conv2d_hwcn.cuda",
            )
        elif layout == "NHWC":
            assert kernel_layout == "HWIO"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.cuda.conv2d_nhwc),
                wrap_topi_schedule(topi.cuda.schedule_conv2d_nhwc),
                name="conv2d_nhwc.cuda",
            )

            N, H, W, _ = get_const_tuple(data.shape)
            KH, KW, CI, CO = get_const_tuple(kernel.shape)
            # Winograd shape related judgment
            (
                judge_winograd_tensorcore,
                judge_winograd_autotvm,
                judge_winograd_auto_scheduler,
            ) = judge_winograd(
                N,
                H,
                W,
                KH,
                KW,
                CI,
                CO,
                padding,
                stride_h,
                stride_w,
                dilation_h,
                dilation_w,
                data.dtype,
                kernel.dtype,
                pre_flag=False,
            )
            if judge_winograd_autotvm:
                if (
                    target.kind.name == "cuda"
                    and nvcc.have_tensorcore(target=target)
                    and judge_winograd_tensorcore
                ):
                    strategy.add_implementation(
                        wrap_compute_conv2d(topi.cuda.conv2d_nhwc_winograd_tensorcore),
                        wrap_topi_schedule(topi.cuda.schedule_conv2d_nhwc_winograd_tensorcore),
                        name="conv2d_nhwc_winograd_tensorcore.cuda",
                        plevel=5,
                    )
                else:
                    strategy.add_implementation(
                        wrap_compute_conv2d(topi.cuda.conv2d_nhwc_winograd_direct),
                        wrap_topi_schedule(topi.cuda.schedule_conv2d_nhwc_winograd_direct),
                        name="conv2d_nhwc_winograd_direct.cuda",
                        plevel=5,
                    )
            if (
                target.kind.name == "cuda"
                and nvcc.have_tensorcore(target=target)
                and (
                    (N % 16 == 0 and CI % 16 == 0 and CO % 16 == 0)
                    or (N % 8 == 0 and CI % 16 == 0 and CO % 32 == 0)
                    or (N % 32 == 0 and CI % 16 == 0 and CO % 8 == 0)
                )
            ):
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.cuda.conv2d_nhwc_tensorcore),
                    wrap_topi_schedule(topi.cuda.schedule_conv2d_nhwc_tensorcore),
                    name="conv2d_nhwc_tensorcore.cuda",
                    plevel=20,
                )

            # register auto-scheduler implementations
            if is_auto_scheduler_enabled() and judge_winograd_auto_scheduler:
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.nn.conv2d_winograd_nhwc),
                    naive_schedule,  # this implementation should never be picked by autotvm
                    name="conv2d_nhwc.winograd",
                    plevel=15,
                )

        elif layout == "HWNC":
            assert kernel_layout in ["HWOI", "HWOI16o16i", "HWOI8o32i", "HWOI32o16i"]
            _, _, N, in_channels = get_const_tuple(data.shape)
            pre_computed = len(kernel.shape) == 6
            if pre_computed:
                _, _, oc_chunk, _, oc_block_factor, _ = get_const_tuple(kernel.shape)
                out_channels = oc_chunk * oc_block_factor
            else:
                _, _, out_channels, _ = get_const_tuple(kernel.shape)

            tensorcore_dtypes = ["int4", "uint4", "int8", "uint8"]
            if (
                (N % 16 == 0 and in_channels % 16 == 0 and out_channels % 16 == 0)
                or (N % 8 == 0 and in_channels % 16 == 0 and out_channels % 32 == 0)
                or (N % 32 == 0 and in_channels % 16 == 0 and out_channels % 8 == 0)
                and (data.dtype in tensorcore_dtypes and kernel.dtype in tensorcore_dtypes)
            ):
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.cuda.conv2d_hwnc_tensorcore),
                    wrap_topi_schedule(topi.cuda.schedule_conv2d_hwnc_tensorcore),
                    name="conv2d_hwnc_tensorcore_direct.cuda",
                    plevel=20,
                )
            else:
                raise RuntimeError(
                    "Unsupported shape for conv2d HWNC.\
                                    Need to satisfy tensor core schedule."
                )
        elif layout == "NCHW4c" and data.dtype in ["int8", "uint8"]:
            assert kernel_layout == "OIHW4o4i"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.cuda.conv2d_NCHWc_int8, True),
                wrap_topi_schedule(topi.cuda.schedule_conv2d_NCHWc_int8),
                name="conv2d_NCHWc_int8.cuda",
            )
        else:
            raise RuntimeError("Unsupported conv2d layout {} for CUDA".format(layout))
        # add cudnn implementation
        if target.kind.name == "cuda" and "cudnn" in target.libs:
            if layout in ["NCHW", "NHWC"] and padding[0] == padding[2] and padding[1] == padding[3]:
                strategy.add_implementation(
                    wrap_compute_conv2d(
                        topi.cuda.conv2d_cudnn, need_data_layout=True, has_groups=True
                    ),
                    wrap_topi_schedule(topi.cuda.schedule_conv2d_cudnn),
                    name="conv2d_cudnn.cuda",
                    plevel=25,
                )
    elif is_depthwise_conv2d(data.shape, layout, kernel.shape, kernel_layout, groups):
        if layout == "NCHW":
            assert kernel_layout == "OIHW"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.cuda.depthwise_conv2d_nchw),
                wrap_topi_schedule(topi.cuda.schedule_depthwise_conv2d_nchw),
                name="depthwise_conv2d_nchw.cuda",
            )
        elif layout == "NHWC":
            assert kernel_layout == "HWOI"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.nn.depthwise_conv2d_nhwc),
                wrap_topi_schedule(topi.cuda.schedule_depthwise_conv2d_nhwc),
                name="depthwise_conv2d_nhwc.cuda",
            )
        else:
            raise RuntimeError("Unsupported depthwise_conv2d layout {}".format(layout))
    else:  # group_conv2d
        # add cudnn implementation, if any
        cudnn_impl = False
        if target.kind.name == "cuda" and "cudnn" in target.libs:
            if layout in ["NCHW", "NHWC"] and padding[0] == padding[2] and padding[1] == padding[3]:
                strategy.add_implementation(
                    wrap_compute_conv2d(
                        topi.cuda.conv2d_cudnn, need_data_layout=True, has_groups=True
                    ),
                    wrap_topi_schedule(topi.cuda.schedule_conv2d_cudnn),
                    name="conv2d_cudnn.cuda",
                    plevel=25,
                )
                cudnn_impl = True

        if layout == "NCHW":
            # TODO(@vinx13, @icemelon9): Use group_conv2d_NCHWc_int8 when dtype is int8/uint8.
            assert kernel_layout == "OIHW"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.cuda.group_conv2d_nchw, has_groups=True),
                wrap_topi_schedule(topi.cuda.schedule_group_conv2d_nchw),
                name="group_conv2d_nchw.cuda",
            )
        elif layout == "NCHW4c" and data.dtype in ["int8", "uint8"]:
            assert kernel_layout == "OIHW4o4i"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.cuda.group_conv2d_NCHWc_int8, True),
                wrap_topi_schedule(topi.cuda.schedule_group_conv2d_NCHWc_int8),
                name="group_conv2d_NCHWc_int8.cuda",
            )
        elif not cudnn_impl:
            raise RuntimeError("Unsupported group_conv2d layout {}".format(layout))
    return strategy
