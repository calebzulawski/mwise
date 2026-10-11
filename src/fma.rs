//! Portable SIMD fma with hardware and software implementations.
mod hardware;
mod software;

use core::simd::Simd;

/// Computes `x * y + z`, rounding only once to the result type.
#[inline]
pub fn fma_f32<const N: usize>(x: Simd<f32, N>, y: Simd<f32, N>, z: Simd<f32, N>) -> Simd<f32, N> {
    if crate::backend::use_hardware_fma_f32::<N>() {
        hardware::fma_f32(x, y, z)
    } else {
        software::fma_f32(x, y, z)
    }
}

/// Computes `x * y + z`, rounding only once to the result type.
#[inline]
pub fn fma_f64<const N: usize>(x: Simd<f64, N>, y: Simd<f64, N>, z: Simd<f64, N>) -> Simd<f64, N> {
    if crate::backend::use_hardware_fma_f64::<N>() {
        hardware::fma_f64(x, y, z)
    } else {
        software::fma_f64(x, y, z)
    }
}
