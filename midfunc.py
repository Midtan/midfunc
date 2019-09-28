# Some function(s) I probably used in some way or the other

from functools import partial
import vapoursynth as vs
import fvsfunc as fvf  # https://github.com/Irrational-Encoding-Wizardry/fvsfunc
import nnedi3_rpow2 as rpow2  # https://github.com/darealshinji/vapoursynth-plugins/blob/master/scripts/nnedi3_rpow2.py

# Functions included:
# Descale444ToTarget

# Global requirements:
# fvsfunc: https://github.com/Irrational-Encoding-Wizardry/fvsfunc
# nnedi3_rpow2: https://github.com/darealshinji/vapoursynth-plugins/blob/master/scripts/nnedi3_rpow2.py

core = vs.core;

"""
fvf.DescaleAA() and fvf.Descale[M]() should only be used when the desired resolution is the original resolution
or production resolution.
Use this function if you want to end up on another resolution.
When does this make sense?
- If your source has sections with different production resolutions one of those sections will have to end up on a third resolution

Parameters
- native_kernel: str = 'bicubic - kernel to use to descale to production resolution, 
    as supported by fvf.Resize()
- descale_masked: bool = True - set to false if you don't want a masked descale with fvf.DescaleM()
- native_width: int = None - width of the production resolution
- native_height: int = None - height of the production resolution
- target_kernel: str = spline36 - kernel to use to scale back to target resolution, 
    as supported by fvf.Resize() or rpow2.nnedi3rpow2()
- target_width: int = None - width of the target resolution
- target_height: int = None - height of the target resolution
- nnedi3rpow2: bool = False - if set to True it will use nnedi3rpow2 if target resolution is higher than production resolution
- **kwargs - parameters as specified in fvf.DescaleM() or fvf.Resize() depending on the value of descale_masked
"""


def Descale444ToTarget(clip: vs.clipNode, native_kernel: str = 'bicubic', descale_masked: bool = True,
                       native_width: int = None, native_height: int = None, target_kernel: str = 'spline36',
                       target_width: int = None, target_height: int = None, nnedi3_rpow2: bool = False, **kwargs):
    if native_width is None:
        raise ValueError('Descale444ToTarget: parameter "native_width" is required!')
    if native_height is None:
        raise ValueError('Descale444ToTarget: parameter "native_height" is required!')
    if target_width is None:
        raise ValueError('Descale444ToTarget: parameter "target_width" is required!')
    if target_height is None:
        raise ValueError('Descale444ToTarget: parameter "target_height" is required!')

    y = clip.std.ShufflePlanes(planes=0, colorfamily=vs.GRAY)
    u = clip.std.ShufflePlanes(planes=1, colorfamily=vs.GRAY)
    v = clip.std.ShufflePlanes(planes=2, colorfamily=vs.GRAY)

    if descale_masked:
        y = fvf.DescaleM(y, descale_kernel=native_kernel, w=native_width, h=native_height, **kwargs)
    else:
        y = fvf.Resize(y, kernel=native_kernel, w=native_width, h=native_height, invks=True, **kwargs)

    if y.height < target_height and y.width < target_width and y.height / y.width == target_height / target_width and nnedi3_rpow2:
        y = rpow2.nnedi3_rpow2(y, correct_shift=True, kernel=target_kernel, width=target_width, height=target_height)
    elif (native_height, native_width) != (target_height, target_width) or not nnedi3_rpow2:
        y = fvf.Resize(y, w=target_width, h=target_height, kernel=target_kernel)

    if u.height < target_height and u.width < target_width and u.height / u.width == target_height / target_width:
        u = rpow2.nnedi3_rpow2(u, correct_shift=True, kernel=target_kernel, width=y.width, height=y.height)
        v = rpow2.nnedi3_rpow2(v, correct_shift=True, kernel=target_kernel, width=y.width, height=y.height)
    elif (u.height, u.width) != (target_height, target_width):
        u = fvf.Resize(u, w=y.width, h=y.height, kernel=target_kernel, sx=0.25)
        v = fvf.Resize(v, w=y.width, h=y.height, kernel=target_kernel, sx=0.25)

    clip = core.std.ShufflePlanes(clips=[y, u, v], planes=[0, 0, 0], colorfamily=vs.YUV)

    return clip


DescaleM444ToTarget = partial(Descale444ToTarget, descale_masked=True)

Debilinear444ToTarget = partial(Descale444ToTarget, native_kernel='bilinear')
DebilinearM444ToTarget = partial(Descale444ToTarget, native_kernel='bilinear', descale_masked=True)

Debicubic444ToTarget = partial(Descale444ToTarget, native_kernel='bicubic')
DebicubicM444ToTarget = partial(Descale444ToTarget, native_kernel='bicubic', descale_masked=True)

Delanczos444ToTarget = partial(Descale444ToTarget, native_kernel='lanczos')
DelanczosM444ToTarget = partial(Descale444ToTarget, native_kernel='lanczos', descale_masked=True)

Despline16444ToTarget = partial(Descale444ToTarget, native_kernel='spline16')
Despline16M444ToTarget = partial(Descale444ToTarget, native_kernel='spline16', descale_masked=True)

Despline36444ToTarget = partial(Descale444ToTarget, native_kernel='spline36')
Despline36M444ToTarget = partial(Descale444ToTarget, native_kernel='spline36', descale_masked=True)
