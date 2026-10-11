//! Portable SIMD trigonometric functions and their shared kernels.
pub(crate) mod data;
pub(crate) mod inverse_fma;
pub(crate) mod inverse_non_fma;
pub(crate) mod reduction;
pub(crate) mod sin_cos_fma;
pub(crate) mod sin_cos_non_fma;
pub(crate) mod sin_cos_non_fma_fast;
pub(crate) mod sin_cos_pair_fma;
pub(crate) mod sin_cos_pair_non_fma;
pub(crate) mod tan;
pub(crate) mod tan_fma;

use core::simd::Simd;

/// Computes the sine of the radian angle `x`.
///
/// Accuracy target: at most 4 ULP.
#[inline]
pub fn sin_f32<const N: usize>(x: Simd<f32, N>) -> Simd<f32, N> {
    if crate::backend::use_hardware_fma_f32() {
        sin_cos_fma::sin_cos_f32::<N, false>(x)
    } else {
        sin_cos_non_fma_fast::sin_cos_f32::<N, false>(x)
    }
}

/// Computes the sine of the radian angle `x`.
///
/// Accuracy target: at most 4 ULP.
#[inline]
pub fn sin_f64<const N: usize>(x: Simd<f64, N>) -> Simd<f64, N> {
    if crate::backend::use_hardware_fma_f64() {
        sin_cos_fma::sin_cos_f64::<N, false>(x)
    } else {
        sin_cos_non_fma::full_range_sincos_f64(x).0
    }
}

/// Computes the cosine of the radian angle `x`.
///
/// Accuracy target: at most 4 ULP.
#[inline]
pub fn cos_f32<const N: usize>(x: Simd<f32, N>) -> Simd<f32, N> {
    if crate::backend::use_hardware_fma_f32() {
        sin_cos_fma::sin_cos_f32::<N, true>(x)
    } else {
        sin_cos_non_fma_fast::sin_cos_f32::<N, true>(x)
    }
}

/// Computes the cosine of the radian angle `x`.
///
/// Accuracy target: at most 4 ULP.
#[inline]
pub fn cos_f64<const N: usize>(x: Simd<f64, N>) -> Simd<f64, N> {
    if crate::backend::use_hardware_fma_f64() {
        sin_cos_fma::sin_cos_f64::<N, true>(x)
    } else {
        sin_cos_non_fma::full_range_sincos_f64(x).1
    }
}

/// Computes the sine and cosine of the radian angle `x`, returning `(sin, cos)`.
///
/// Accuracy target: at most 4 ULP.
#[inline]
pub fn sincos_f32<const N: usize>(x: Simd<f32, N>) -> (Simd<f32, N>, Simd<f32, N>) {
    if crate::backend::use_hardware_fma_f32() {
        sin_cos_pair_fma::sincos_f32(x)
    } else {
        sin_cos_pair_non_fma::sincos_f32(x)
    }
}

/// Computes the sine and cosine of the radian angle `x`, returning `(sin, cos)`.
///
/// Accuracy target: at most 4 ULP.
#[inline]
pub fn sincos_f64<const N: usize>(x: Simd<f64, N>) -> (Simd<f64, N>, Simd<f64, N>) {
    if crate::backend::use_hardware_fma_f64() {
        sin_cos_pair_fma::sincos_f64(x)
    } else {
        sin_cos_non_fma::full_range_sincos_f64(x)
    }
}

/// Computes the tangent of the radian angle `x`.
///
/// Accuracy target: at most 4 ULP.
#[inline]
pub fn tan_f32<const N: usize>(x: Simd<f32, N>) -> Simd<f32, N> {
    if crate::backend::use_hardware_fma_f32() {
        tan_fma::tan_f32(x)
    } else {
        tan::tan_f32(x)
    }
}

/// Computes the tangent of the radian angle `x`.
///
/// Accuracy target: at most 4 ULP.
#[inline]
pub fn tan_f64<const N: usize>(x: Simd<f64, N>) -> Simd<f64, N> {
    if crate::backend::use_hardware_fma_f64() {
        tan_fma::tan_f64(x)
    } else {
        tan::tan_f64(x)
    }
}

/// Computes the principal arc sine of `x`, in radians.
///
/// Accuracy target: at most 4 ULP.
#[inline]
pub fn asin_f32<const N: usize>(x: Simd<f32, N>) -> Simd<f32, N> {
    if crate::backend::use_hardware_fma_f32() {
        inverse_fma::asin_f32(x)
    } else {
        inverse_non_fma::asin_f32(x)
    }
}

/// Computes the principal arc sine of `x`, in radians.
///
/// Accuracy target: at most 4 ULP.
#[inline]
pub fn asin_f64<const N: usize>(x: Simd<f64, N>) -> Simd<f64, N> {
    if crate::backend::use_hardware_fma_f64() {
        inverse_fma::asin_f64(x)
    } else {
        inverse_non_fma::asin_f64(x)
    }
}

/// Computes the principal arc cosine of `x`, in radians.
///
/// Accuracy target: at most 4 ULP.
#[inline]
pub fn acos_f32<const N: usize>(x: Simd<f32, N>) -> Simd<f32, N> {
    if crate::backend::use_hardware_fma_f32() {
        inverse_fma::acos_f32(x)
    } else {
        inverse_non_fma::acos_f32(x)
    }
}

/// Computes the principal arc cosine of `x`, in radians.
///
/// Accuracy target: at most 4 ULP.
#[inline]
pub fn acos_f64<const N: usize>(x: Simd<f64, N>) -> Simd<f64, N> {
    if crate::backend::use_hardware_fma_f64() {
        inverse_fma::acos_f64(x)
    } else {
        inverse_non_fma::acos_f64(x)
    }
}

/// Computes the principal arc tangent of `x`, in radians.
///
/// Accuracy target: at most 4 ULP.
#[inline]
pub fn atan_f32<const N: usize>(x: Simd<f32, N>) -> Simd<f32, N> {
    if crate::backend::use_hardware_fma_f32() {
        inverse_fma::atan_f32(x)
    } else {
        inverse_non_fma::atan_f32(x)
    }
}

/// Computes the principal arc tangent of `x`, in radians.
///
/// Accuracy target: at most 4 ULP.
#[inline]
pub fn atan_f64<const N: usize>(x: Simd<f64, N>) -> Simd<f64, N> {
    if crate::backend::use_hardware_fma_f64() {
        inverse_fma::atan_f64(x)
    } else {
        inverse_non_fma::atan_f64(x)
    }
}

/// Computes the arc tangent of `y / x`, in radians, using both signs to determine the quadrant.
///
/// Accuracy target: at most 4 ULP.
#[inline]
pub fn atan2_f32<const N: usize>(y: Simd<f32, N>, x: Simd<f32, N>) -> Simd<f32, N> {
    if crate::backend::use_hardware_fma_f32() {
        inverse_fma::atan2_f32(y, x)
    } else {
        inverse_non_fma::atan2_f32(y, x)
    }
}

/// Computes the arc tangent of `y / x`, in radians, using both signs to determine the quadrant.
///
/// Accuracy target: at most 4 ULP.
#[inline]
pub fn atan2_f64<const N: usize>(y: Simd<f64, N>, x: Simd<f64, N>) -> Simd<f64, N> {
    if crate::backend::use_hardware_fma_f64() {
        inverse_fma::atan2_f64(y, x)
    } else {
        inverse_non_fma::atan2_f64(y, x)
    }
}
