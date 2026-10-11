//! Portable SIMD rounding.
mod hardware;
mod software;

use core::simd::{Simd, num::SimdFloat};

use crate::backend::RoundingMode;

/// Rounds `x` to an integer value toward zero.
#[inline]
pub fn trunc_f32<const N: usize>(x: Simd<f32, N>) -> Simd<f32, N> {
    if crate::backend::use_hardware_round_f32(RoundingMode::Directed) {
        hardware::trunc_f32(x)
    } else {
        software::trunc_f32(x)
    }
}

/// Rounds `x` to an integer value toward zero.
#[inline]
pub fn trunc_f64<const N: usize>(x: Simd<f64, N>) -> Simd<f64, N> {
    if crate::backend::use_hardware_round_f64(RoundingMode::Directed) {
        hardware::trunc_f64(x)
    } else {
        software::trunc_f64(x)
    }
}

/// Returns the greatest integer value less than or equal to `x`.
#[inline]
pub fn floor_f32<const N: usize>(x: Simd<f32, N>) -> Simd<f32, N> {
    if crate::backend::use_hardware_round_f32(RoundingMode::Directed) {
        hardware::floor_f32(x)
    } else {
        software::floor_f32(x)
    }
}

/// Returns the greatest integer value less than or equal to `x`.
#[inline]
pub fn floor_f64<const N: usize>(x: Simd<f64, N>) -> Simd<f64, N> {
    if crate::backend::use_hardware_round_f64(RoundingMode::Directed) {
        hardware::floor_f64(x)
    } else {
        software::floor_f64(x)
    }
}

/// Returns the least integer value greater than or equal to `x`.
#[inline]
pub fn ceil_f32<const N: usize>(x: Simd<f32, N>) -> Simd<f32, N> {
    if crate::backend::use_hardware_round_f32(RoundingMode::Directed) {
        hardware::ceil_f32(x)
    } else {
        software::ceil_f32(x)
    }
}

/// Returns the least integer value greater than or equal to `x`.
#[inline]
pub fn ceil_f64<const N: usize>(x: Simd<f64, N>) -> Simd<f64, N> {
    if crate::backend::use_hardware_round_f64(RoundingMode::Directed) {
        hardware::ceil_f64(x)
    } else {
        software::ceil_f64(x)
    }
}

/// Rounds `x` to the nearest integer value, with ties away from zero.
#[inline]
pub fn round_f32<const N: usize>(x: Simd<f32, N>) -> Simd<f32, N> {
    if crate::backend::use_hardware_round_f32(RoundingMode::TiesAway) {
        hardware::round_f32(x)
    } else {
        software::round_f32(x)
    }
}

/// Rounds `x` to the nearest integer value, with ties away from zero.
#[inline]
pub fn round_f64<const N: usize>(x: Simd<f64, N>) -> Simd<f64, N> {
    if crate::backend::use_hardware_round_f64(RoundingMode::TiesAway) {
        hardware::round_f64(x)
    } else {
        software::round_f64(x)
    }
}

/// Rounds `x` to the nearest integer value, with ties to even.
#[inline]
pub fn roundeven_f32<const N: usize>(x: Simd<f32, N>) -> Simd<f32, N> {
    if crate::backend::use_hardware_round_f32(RoundingMode::TiesEven) {
        hardware::roundeven_f32(x)
    } else {
        software::roundeven_f32(x)
    }
}

/// Rounds `x` to the nearest integer value, with ties to even.
#[inline]
pub fn roundeven_f64<const N: usize>(x: Simd<f64, N>) -> Simd<f64, N> {
    if crate::backend::use_hardware_round_f64(RoundingMode::TiesEven) {
        hardware::roundeven_f64(x)
    } else {
        software::roundeven_f64(x)
    }
}

/// Rounds `x` to the nearest integer, with ties away from zero.
///
/// Results outside the integer range (including infinities) saturate to
/// `i32::MIN` or `i32::MAX`; NaNs produce zero.
#[inline]
pub fn lround_f32<const N: usize>(x: Simd<f32, N>) -> Simd<i32, N> {
    round_f32(x).cast()
}

/// Rounds `x` to the nearest integer, with ties away from zero.
///
/// Results outside the integer range (including infinities) saturate to
/// `i32::MIN` or `i32::MAX`; NaNs produce zero.
#[inline]
pub fn lround_f64<const N: usize>(x: Simd<f64, N>) -> Simd<i32, N> {
    round_f64(x).cast()
}

/// Rounds `x` to the nearest integer, with ties away from zero.
///
/// Results outside the integer range (including infinities) saturate to
/// `i64::MIN` or `i64::MAX`; NaNs produce zero.
#[inline]
pub fn llround_f32<const N: usize>(x: Simd<f32, N>) -> Simd<i64, N> {
    round_f32(x).cast()
}

/// Rounds `x` to the nearest integer, with ties away from zero.
///
/// Results outside the integer range (including infinities) saturate to
/// `i64::MIN` or `i64::MAX`; NaNs produce zero.
#[inline]
pub fn llround_f64<const N: usize>(x: Simd<f64, N>) -> Simd<i64, N> {
    round_f64(x).cast()
}
