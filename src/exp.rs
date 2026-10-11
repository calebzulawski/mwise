//! Portable SIMD exponentials.
//! Implementation selection is confined to these public entry points.
pub(crate) mod data;
pub(crate) mod fma;
pub(crate) mod full_range;
pub(crate) mod non_fma;

use core::simd::Simd;

/// Computes the base-e exponential of `x`.
///
/// Accuracy target: at most 4 ULP.
#[inline]
pub fn exp_f32<const N: usize>(x: Simd<f32, N>) -> Simd<f32, N> {
    if crate::backend::use_hardware_fma_f32() {
        fma::exp_f32(x)
    } else {
        non_fma::exp_f32(x)
    }
}

/// Computes the base-e exponential of `x`.
///
/// Accuracy target: at most 4 ULP.
#[inline]
pub fn exp_f64<const N: usize>(x: Simd<f64, N>) -> Simd<f64, N> {
    if crate::backend::use_hardware_fma_f64() {
        fma::exp_f64(x)
    } else {
        non_fma::exp_f64(x)
    }
}

/// Computes the base-2 exponential of `x`.
///
/// Accuracy target: at most 4 ULP.
#[inline]
pub fn exp2_f32<const N: usize>(x: Simd<f32, N>) -> Simd<f32, N> {
    if crate::backend::use_hardware_fma_f32() {
        fma::exp2_f32(x)
    } else {
        non_fma::exp2_f32(x)
    }
}

/// Computes the base-2 exponential of `x`.
///
/// Accuracy target: at most 4 ULP.
#[inline]
pub fn exp2_f64<const N: usize>(x: Simd<f64, N>) -> Simd<f64, N> {
    if crate::backend::use_hardware_fma_f64() {
        fma::exp2_f64(x)
    } else {
        non_fma::exp2_f64(x)
    }
}
