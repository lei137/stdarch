//! PowerPC Vector Scalar eXtensions (VSX) intrinsics.
//!
//! The references are: [POWER ISA v2.07B (for POWER8 & POWER8 with NVIDIA
//! NVlink)] and [POWER ISA v3.0B (for POWER9)].
//!
//! [POWER ISA v2.07B (for POWER8 & POWER8 with NVIDIA NVlink)]: https://ibm.box.com/s/jd5w15gz301s5b5dt375mshpq9c3lh4u
//! [POWER ISA v3.0B (for POWER9)]: https://ibm.box.com/s/1hzcwkwf8rbju5h9iyf44wm94amnlcrv

#![allow(non_camel_case_types)]

use crate::core_arch::powerpc::*;
use crate::core_arch::simd::*;
use super::macros::*;
use crate::intrinsics::simd::*;

#[cfg(test)]
use stdarch_test::assert_instr;

use crate::mem::{self, transmute};

types! {
    #![unstable(feature = "stdarch_powerpc", issue = "111145")]

    // pub struct vector_Float16 = f16x8;
    /// PowerPC-specific 128-bit wide vector of two packed `i64`
    pub struct vector_signed_long(2 x i64);
    /// PowerPC-specific 128-bit wide vector of two packed `u64`
    pub struct vector_unsigned_long(2 x u64);
    /// PowerPC-specific 128-bit wide vector mask of two `i64`
    pub struct vector_bool_long(2 x i64);
    /// PowerPC-specific 128-bit wide vector of two packed `f64`
    pub struct vector_double(2 x f64);
    // pub struct vector_signed_long_long = vector_signed_long;
    // pub struct vector_unsigned_long_long = vector_unsigned_long;
    // pub struct vector_bool_long_long = vector_bool_long;
    // pub struct vector_signed___int128 = i128x1;
    // pub struct vector_unsigned___int128 = i128x1;
}

#[unstable(feature = "stdarch_powerpc", issue = "111145")]
impl From<m64x2> for vector_bool_long {
    #[inline]
    fn from(value: m64x2) -> Self {
        unsafe { transmute(value) }
    }
}

#[unstable(feature = "stdarch_powerpc", issue = "111145")]
impl From<vector_bool_long> for m64x2 {
    #[inline]
    fn from(value: vector_bool_long) -> Self {
        unsafe { transmute(value) }
    }
}

#[allow(improper_ctypes)]
unsafe extern "unadjusted" {
    #[link_name = "llvm.ppc.altivec.vperm"]
    fn vperm(
        a: vector_signed_int,
        b: vector_signed_int,
        c: vector_unsigned_char,
    ) -> vector_signed_int;
}

// Implement vec_xl for f64 (vector_double) - VSX extension
// Use the impl_vec_xl macro from macros module
use crate::core_arch::powerpc::altivec::sealed::VectorXl;
impl_vec_xl! { vec_xl_f64 lxvd2x / lxv f64 }

mod sealed {
    use super::*;

    #[unstable(feature = "stdarch_powerpc", issue = "111145")]
    pub trait VectorPermDI {
        #[unstable(feature = "stdarch_powerpc", issue = "111145")]
        unsafe fn vec_xxpermdi(self, b: Self, dm: u8) -> Self;
    }

    // xxpermdi has an big-endian bias and extended mnemonics
    #[inline]
    #[target_feature(enable = "vsx")]
    #[cfg_attr(all(test, target_endian = "little"), assert_instr(xxmrgld, dm = 0x0))]
    #[cfg_attr(all(test, target_endian = "big"), assert_instr(xxspltd, dm = 0x0))]
    unsafe fn xxpermdi(a: vector_signed_long, b: vector_signed_long, dm: u8) -> vector_signed_long {
        let a: i64x2 = transmute(a);
        let b: i64x2 = transmute(b);
        let r: i64x2 = match dm & 0b11 {
            0 => simd_shuffle!(a, b, [0b00, 0b10]),
            1 => simd_shuffle!(a, b, [0b01, 0b10]),
            2 => simd_shuffle!(a, b, [0b00, 0b11]),
            _ => simd_shuffle!(a, b, [0b01, 0b11]),
        };
        transmute(r)
    }

    macro_rules! vec_xxpermdi {
        {$impl: ident} => {
            #[unstable(feature = "stdarch_powerpc", issue = "111145")]
            impl VectorPermDI for $impl {
                #[inline]
                #[target_feature(enable = "vsx")]
                unsafe fn vec_xxpermdi(self, b: Self, dm: u8) -> Self {
                    transmute(xxpermdi(transmute(self), transmute(b), dm))
                }
            }
        }
    }

    vec_xxpermdi! { vector_unsigned_long }
    vec_xxpermdi! { vector_signed_long }
    vec_xxpermdi! { vector_bool_long }
    vec_xxpermdi! { vector_double }

    #[unstable(feature = "stdarch_powerpc", issue = "111145")]
    pub trait VectorMergeEo {
        #[unstable(feature = "stdarch_powerpc", issue = "111145")]
        unsafe fn vec_mergee(self, b: Self) -> Self;
        #[unstable(feature = "stdarch_powerpc", issue = "111145")]
        unsafe fn vec_mergeo(self, b: Self) -> Self;
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(
        all(test, target_endian = "little", target_feature = "power8-vector"),
        assert_instr(vmrgow)
    )]
    #[cfg_attr(
        all(test, target_endian = "big", target_feature = "power8-vector"),
        assert_instr(vmrgew)
    )]
    unsafe fn mergee(a: vector_signed_int, b: vector_signed_int) -> vector_signed_int {
        let p = transmute(u8x16::new(
            0x00, 0x01, 0x02, 0x03, 0x10, 0x11, 0x12, 0x13, 0x08, 0x09, 0x0A, 0x0B, 0x18, 0x19,
            0x1A, 0x1B,
        ));
        vec_perm(a, b, p)
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(
        all(test, target_endian = "little", target_feature = "power8-vector"),
        assert_instr(vmrgew)
    )]
    #[cfg_attr(
        all(test, target_endian = "big", target_feature = "power8-vector"),
        assert_instr(vmrgow)
    )]
    unsafe fn mergeo(a: vector_signed_int, b: vector_signed_int) -> vector_signed_int {
        let p = transmute(u8x16::new(
            0x04, 0x05, 0x06, 0x07, 0x14, 0x15, 0x16, 0x17, 0x0C, 0x0D, 0x0E, 0x0F, 0x1C, 0x1D,
            0x1E, 0x1F,
        ));
        vec_perm(a, b, p)
    }

    macro_rules! vec_mergeeo {
        { $impl: ident, $even: ident, $odd: ident } => {
            #[unstable(feature = "stdarch_powerpc", issue = "111145")]
            impl VectorMergeEo for $impl {
                #[inline]
                #[target_feature(enable = "altivec")]
                unsafe fn vec_mergee(self, b: Self) -> Self {
                    transmute(mergee(transmute(self), transmute(b)))
                }
                #[inline]
                #[target_feature(enable = "altivec")]
                unsafe fn vec_mergeo(self, b: Self) -> Self {
                    transmute(mergeo(transmute(self), transmute(b)))
                }
            }
        }
    }

    vec_mergeeo! { vector_signed_int, mergee, mergeo }
    vec_mergeeo! { vector_unsigned_int, mergee, mergeo }
    vec_mergeeo! { vector_bool_int, mergee, mergeo }
    vec_mergeeo! { vector_float, mergee, mergeo }

    #[inline]
    #[target_feature(enable = "vsx")]
    #[cfg_attr(test, assert_instr(xvadddp))]
    pub(crate) unsafe fn vec_add_double_double(
        a: vector_double,
        b: vector_double,
    ) -> vector_double {
        simd_add(a, b)
    }

    #[inline]
    #[target_feature(enable = "vsx")]
    #[cfg_attr(test, assert_instr(xvsubdp))]
    pub(crate) unsafe fn vec_sub_double_double(
        a: vector_double,
        b: vector_double,
    ) -> vector_double {
        simd_sub(a, b)
    }

    #[inline]
    #[target_feature(enable = "vsx")]
    #[cfg_attr(test, assert_instr(xvmuldp))]
    pub(crate) unsafe fn vec_mul_double_double(
        a: vector_double,
        b: vector_double,
    ) -> vector_double {
        simd_mul(a, b)
    }
}

// Implement AltiVec's VectorAdd trait for vector_double to enable vec_add support
#[unstable(feature = "stdarch_powerpc", issue = "111145")]
impl crate::core_arch::powerpc::altivec::sealed::VectorAdd<vector_double> for vector_double {
    type Result = vector_double;
    #[inline]
    #[target_feature(enable = "vsx")]
    unsafe fn vec_add(self, other: vector_double) -> Self::Result {
        sealed::vec_add_double_double(self, other)
    }
}

// Implement AltiVec's VectorSub trait for vector_double to enable vec_sub support
#[unstable(feature = "stdarch_powerpc", issue = "111145")]
impl crate::core_arch::powerpc::altivec::sealed::VectorSub<vector_double> for vector_double {
    type Result = vector_double;
    #[inline]
    #[target_feature(enable = "vsx")]
    unsafe fn vec_sub(self, other: vector_double) -> Self::Result {
        sealed::vec_sub_double_double(self, other)
    }
}

// Implement AltiVec's VectorMul trait for vector_double to enable vec_mul support.
#[unstable(feature = "stdarch_powerpc", issue = "111145")]
impl crate::core_arch::powerpc::altivec::sealed::VectorMul for vector_double {
    #[inline]
    #[target_feature(enable = "vsx")]
    unsafe fn vec_mul(self, b: Self) -> Self {
        sealed::vec_mul_double_double(self, b)
    }
}

// Macro to implement VectorCmpEq trait for vector types.
macro_rules! impl_vec_cmpeq {
    ($vec_ty:ident, $result_ty:ident, $mask_ty:ident, $instr:ident) => {
        #[unstable(feature = "stdarch_powerpc", issue = "111145")]
        impl crate::core_arch::powerpc::altivec::sealed::VectorCmpEq<$vec_ty> for $vec_ty {
            type Result = $result_ty;
            #[inline]
            #[target_feature(enable = "vsx")]
            #[cfg_attr(test, assert_instr($instr))]
            unsafe fn vec_cmpeq(self, b: $vec_ty) -> Self::Result {
                let result: $mask_ty = simd_eq(self, b);
                transmute(result)
            }
        }
    };
}
impl_vec_cmpeq!(vector_float, vector_bool_int, m32x4, xvcmpeqsp);
impl_vec_cmpeq!(vector_double, vector_bool_long, m64x2, xvcmpeqdp);

// Macro to implement VectorCmpGt trait for vector types.
macro_rules! impl_vec_cmpgt {
    ($vec_ty:ident, $result_ty:ident, $mask_ty:ident, $instr:ident) => {
        #[unstable(feature = "stdarch_powerpc", issue = "111145")]
        impl crate::core_arch::powerpc::altivec::sealed::VectorCmpGt<$vec_ty> for $vec_ty {
            type Result = $result_ty;
            #[inline]
            #[target_feature(enable = "vsx")]
            #[cfg_attr(test, assert_instr($instr))]
            unsafe fn vec_cmpgt(self, b: $vec_ty) -> Self::Result {
                let result: $mask_ty = simd_gt(self, b);
                transmute(result)
            }
        }
    };
}
impl_vec_cmpgt!(vector_float, vector_bool_int, m32x4, xvcmpgtsp);
impl_vec_cmpgt!(vector_double, vector_bool_long, m64x2, xvcmpgtdp);

// Macro to implement VectorCmpGe trait for vector types.
macro_rules! impl_vec_cmpge {
    ($vec_ty:ident, $result_ty:ident, $mask_ty:ident, $instr:ident) => {
        #[cfg(target_feature = "vsx")]
        #[unstable(feature = "stdarch_powerpc", issue = "111145")]
        impl crate::core_arch::powerpc::altivec::sealed::VectorCmpGe<$vec_ty> for $vec_ty {
            type Result = $result_ty;
            #[inline]
            #[target_feature(enable = "vsx")]
            #[cfg_attr(test, assert_instr($instr))]
            unsafe fn vec_cmpge(self, b: $vec_ty) -> Self::Result {
                let result: $mask_ty = simd_ge(self, b);
                transmute(result)
            }
        }
    };
}

impl_vec_cmpge!(vector_float, vector_bool_int, m32x4, xvcmpgesp);
impl_vec_cmpge!(vector_double, vector_bool_long, m64x2, xvcmpgedp);

// Implement AltiVec's VectorSld trait for vector_double to enable vec_sld support.
use crate::core_arch::powerpc::altivec::sealed::{VectorSld, vsldoi, xxsldwi};
impl_vec_sld! { vector_double }

/// Vector permute.
#[inline]
#[target_feature(enable = "vsx")]
//#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_powerpc", issue = "111145")]
pub unsafe fn vec_xxpermdi<T, const DM: i32>(a: T, b: T) -> T
where
    T: sealed::VectorPermDI,
{
    static_assert_uimm_bits!(DM, 2);
    a.vec_xxpermdi(b, DM as u8)
}

/// Vector Merge Even
///
/// ## Purpose
/// Merges the even-numbered values from two vectors.
///
/// ## Result value
/// The even-numbered elements of a are stored into the even-numbered elements of r.
/// The even-numbered elements of b are stored into the odd-numbered elements of r.
#[inline]
#[target_feature(enable = "altivec")]
#[unstable(feature = "stdarch_powerpc", issue = "111145")]
pub unsafe fn vec_mergee<T>(a: T, b: T) -> T
where
    T: sealed::VectorMergeEo,
{
    a.vec_mergee(b)
}

/// Vector Merge Odd
///
/// ## Purpose
/// Merges the odd-numbered values from two vectors.
///
/// ## Result value
/// The odd-numbered elements of a are stored into the even-numbered elements of r.
/// The odd-numbered elements of b are stored into the odd-numbered elements of r.
#[inline]
#[target_feature(enable = "altivec")]
#[unstable(feature = "stdarch_powerpc", issue = "111145")]
pub unsafe fn vec_mergeo<T>(a: T, b: T) -> T
where
    T: sealed::VectorMergeEo,
{
    a.vec_mergeo(b)
}

#[cfg(test)]
mod tests {
    #[cfg(target_arch = "powerpc")]
    use crate::core_arch::arch::powerpc::*;

    #[cfg(target_arch = "powerpc64")]
    use crate::core_arch::arch::powerpc64::*;

    use crate::core_arch::simd::*;
    use crate::mem::transmute;
    use stdarch_test::simd_test;

    macro_rules! test_vec_xxpermdi {
        {$name:ident, $shorttype:ident, $longtype:ident, [$($a:expr),+], [$($b:expr),+], [$($c:expr),+], [$($d:expr),+]} => {
            #[simd_test(enable = "vsx")]
            fn $name() {
                let a = $longtype::from($shorttype::from_array([$($a),+, $($b),+]));
                let b = $longtype::from($shorttype::from_array([$($c),+, $($d),+]));

                unsafe {
                    assert_eq!($shorttype::from_array([$($a),+, $($c),+]), $shorttype::from(vec_xxpermdi::<_, 0>(a, b)));
                    assert_eq!($shorttype::from_array([$($b),+, $($c),+]), $shorttype::from(vec_xxpermdi::<_, 1>(a, b)));
                    assert_eq!($shorttype::from_array([$($a),+, $($d),+]), $shorttype::from(vec_xxpermdi::<_, 2>(a, b)));
                    assert_eq!($shorttype::from_array([$($b),+, $($d),+]), $shorttype::from(vec_xxpermdi::<_, 3>(a, b)));
                }
            }
        }
    }

    test_vec_xxpermdi! {test_vec_xxpermdi_u64x2, u64x2, vector_unsigned_long, [0], [1], [2], [3]}
    test_vec_xxpermdi! {test_vec_xxpermdi_i64x2, i64x2, vector_signed_long, [0], [-1], [2], [-3]}
    test_vec_xxpermdi! {test_vec_xxpermdi_m64x2, m64x2, vector_bool_long, [false], [true], [false], [true]}
    test_vec_xxpermdi! {test_vec_xxpermdi_f64x2, f64x2, vector_double, [0.0], [1.0], [2.0], [3.0]}

    #[simd_test(enable = "vsx")]
    fn test_vec_add_f64x2_f64x2() {
        let a = vector_double::from(f64x2::from_array([1.0, 2.0]));
        let b = vector_double::from(f64x2::from_array([3.0, 4.0]));
        let expected = vector_double::from(f64x2::from_array([4.0, 6.0]));

        unsafe {
            assert_eq!(f64x2::from(vec_add(a, b)), f64x2::from(expected));
        }
    }
    #[simd_test(enable = "vsx")]
    fn test_vec_sub_f64x2_f64x2() {
        let a = vector_double::from(f64x2::from_array([5.0, 8.0]));
        let b = vector_double::from(f64x2::from_array([3.0, 4.0]));
        let expected = vector_double::from(f64x2::from_array([2.0, 4.0]));

        unsafe {
            assert_eq!(f64x2::from(vec_sub(a, b)), f64x2::from(expected));
        }
    }

    #[simd_test(enable = "vsx")]
    fn test_vec_mul_f64x2_f64x2() {
        let a = vector_double::from(f64x2::from_array([2.0, 3.0]));
        let b = vector_double::from(f64x2::from_array([4.0, 5.0]));
        let expected = vector_double::from(f64x2::from_array([8.0, 15.0]));

        unsafe {
            assert_eq!(f64x2::from(vec_mul(a, b)), f64x2::from(expected));
        }
    }

    #[simd_test(enable = "altivec")]
    fn test_vec_sld_f64x2() {
        use crate::core_arch::powerpc::altivec::sealed::VectorSld;

        let a = vector_double::from(f64x2::from_array([1.0, 2.0]));
        let b = vector_double::from(f64x2::from_array([3.0, 4.0]));

        // Shift left by 8 bytes (1 f64 element)
        // On little-endian: shifts right in memory, result is [b[1], a[0]] = [4.0, 1.0]
        // On big-endian: shifts left in memory, result is [a[1], b[0]] = [2.0, 3.0]
        unsafe {
            let result: f64x2 = transmute(a.vec_sld::<8>(b));
            #[cfg(target_endian = "little")]
            let expected = f64x2::from_array([4.0, 1.0]);
            #[cfg(target_endian = "big")]
            let expected = f64x2::from_array([2.0, 3.0]);
            assert_eq!(result, expected);
        }
    }

    #[simd_test(enable = "vsx")]
    fn test_vec_cmpeq_f32x4() {
        let a = vector_float::from(f32x4::from_array([1.0, 2.0, 3.0, 4.0]));
        let b = vector_float::from(f32x4::from_array([1.0, 3.0, 3.0, 5.0]));

        unsafe {
            let result: vector_bool_int = vec_cmpeq(a, b);
            // Elements 0 and 2 are equal, elements 1 and 3 are not equal.
            // Equal elements should have all bits set (-1), non-equal should be 0.
            let result_i32: i32x4 = transmute(result);
            assert_eq!(result_i32.as_array()[0], -1i32);
            assert_eq!(result_i32.as_array()[1], 0i32);
            assert_eq!(result_i32.as_array()[2], -1i32);
            assert_eq!(result_i32.as_array()[3], 0i32);
        }
    }

    #[simd_test(enable = "vsx")]
    fn test_vec_cmpeq_f64x2() {
        let a = vector_double::from(f64x2::from_array([1.0, 2.0]));
        let b = vector_double::from(f64x2::from_array([1.0, 3.0]));

        unsafe {
            let result: vector_bool_long = vec_cmpeq(a, b);
            // First element equal (1.0 == 1.0), second not equal (2.0 != 3.0).
            // Equal elements should have all bits set (-1), non-equal should be 0.
            let result_i64: i64x2 = transmute(result);
            assert_eq!(result_i64.as_array()[0], -1i64);
            assert_eq!(result_i64.as_array()[1], 0i64);
        }
    }

    #[simd_test(enable = "vsx")]
    fn test_vec_cmpgt_f32x4() {
        let a = vector_float::from(f32x4::from_array([1.0, 2.0, 3.0, 4.0]));
        let b = vector_float::from(f32x4::from_array([0.0, 3.0, 3.0, 5.0]));

        unsafe {
            let result: vector_bool_int = vec_cmpgt(a, b);
            // Element 0: 1.0 > 0.0 (true), Element 1: 2.0 > 3.0 (false)
            // Element 2: 3.0 > 3.0 (false), Element 3: 4.0 > 5.0 (false)
            let result_i32: i32x4 = transmute(result);
            assert_eq!(result_i32.as_array()[0], -1i32);
            assert_eq!(result_i32.as_array()[1], 0i32);
            assert_eq!(result_i32.as_array()[2], 0i32);
            assert_eq!(result_i32.as_array()[3], 0i32);
        }
    }

    #[simd_test(enable = "vsx")]
    fn test_vec_cmpgt_f64x2() {
        let a = vector_double::from(f64x2::from_array([2.0, 1.0]));
        let b = vector_double::from(f64x2::from_array([1.0, 3.0]));

        unsafe {
            let result: vector_bool_long = vec_cmpgt(a, b);
            // First element: 2.0 > 1.0 (true), second: 1.0 > 3.0 (false)
            let result_i64: i64x2 = transmute(result);
            assert_eq!(result_i64.as_array()[0], -1i64);
            assert_eq!(result_i64.as_array()[1], 0i64);
        }
    }

    #[simd_test(enable = "vsx")]
    fn test_vec_cmpge_f32x4() {
        let a = vector_float::from(f32x4::from_array([1.0, 2.0, 3.0, 4.0]));
        let b = vector_float::from(f32x4::from_array([0.0, 3.0, 3.0, 5.0]));

        unsafe {
            let result: vector_bool_int = vec_cmpge(a, b);
            // Element 0: 1.0 >= 0.0 (true), Element 1: 2.0 >= 3.0 (false)
            // Element 2: 3.0 >= 3.0 (true), Element 3: 4.0 >= 5.0 (false)
            let result_i32: i32x4 = transmute(result);
            assert_eq!(result_i32.as_array()[0], -1i32);
            assert_eq!(result_i32.as_array()[1], 0i32);
            assert_eq!(result_i32.as_array()[2], -1i32);
            assert_eq!(result_i32.as_array()[3], 0i32);
        }
    }

    #[simd_test(enable = "vsx")]
    fn test_vec_cmpge_f64x2() {
        let a = vector_double::from(f64x2::from_array([2.0, 3.0]));
        let b = vector_double::from(f64x2::from_array([1.0, 3.0]));

        unsafe {
            let result: vector_bool_long = vec_cmpge(a, b);
            // First element: 2.0 >= 1.0 (true), second: 3.0 >= 3.0 (true)
            let result_i64: i64x2 = transmute(result);
            assert_eq!(result_i64.as_array()[0], -1i64);
            assert_eq!(result_i64.as_array()[1], -1i64);
        }
    }

    #[simd_test(enable = "vsx")]
    fn test_vec_xl_f64() {
        let pat = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        // Test loading from aligned offset 0
        unsafe {
            let result = vec_xl(0, pat.as_ptr());
            let result_f64: f64x2 = transmute(result);
            assert_eq!(result_f64.as_array()[0], 1.0);
            assert_eq!(result_f64.as_array()[1], 2.0);
        }

        // Test loading from offset 16 (2 f64 elements = 16 bytes)
        unsafe {
            let result = vec_xl(16, pat.as_ptr());
            let result_f64: f64x2 = transmute(result);
            assert_eq!(result_f64.as_array()[0], 3.0);
            assert_eq!(result_f64.as_array()[1], 4.0);
        }

        // Test loading from offset 32 (4 f64 elements = 32 bytes)
        unsafe {
            let result = vec_xl(32, pat.as_ptr());
            let result_f64: f64x2 = transmute(result);
            assert_eq!(result_f64.as_array()[0], 5.0);
            assert_eq!(result_f64.as_array()[1], 6.0);
        }
    }
}
