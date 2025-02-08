#![allow(unused)]

use std::cmp::PartialEq;
use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::utilities::compare_float;

/// Type alias for points.
pub type Point = Tuple<PointM>;

/// Type alias for vectors.
pub type Vector = Tuple<VectorM>;

/// Convenience function to create a [Point].
pub fn point(x: f64, y: f64, z: f64) -> Point {
    Tuple::new([x, y, z])
}

/// Convenience function to create a [Vector].
pub fn vector(i: f64, j: f64, k: f64) -> Vector {
    Tuple::new([i, j, k])
}

/// Marker indicating that a [Tuple] represents a point.
#[derive(Clone, Copy, Debug)]
pub struct PointM;

/// Marker indicating that a [Tuple] represents a vector.
#[derive(Clone, Copy, Debug)]
pub struct VectorM;

/// Implementation of [Point]s and [Vector]s.
#[derive(Clone, Copy, Debug)]
pub struct Tuple<T> {
    els: [f64; 3],
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Copy> Tuple<T> {
    pub fn new(els: [f64; 3]) -> Self {
        Self {
            els,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Convert a [Tuple] to a row vector, with its fourth element = 0.0.
    pub fn row4(self) -> [f64; 4] {
        [self.els[0], self.els[1], self.els[2], 0.0]
    }

    /// Convert a [Tuple] to a column vector, with its fourth element = 1.0.
    pub fn col4(self) -> [[f64; 1]; 4] {
        [[self.els[0]], [self.els[1]], [self.els[2]], [1.0]]
    }

    /// Maps the operation to each element of the vector
    fn map_elementwise<U: Copy>(self, op: impl Fn(f64) -> f64) -> Tuple<U> {
        Tuple {
            els: [op(self.els[0]), op(self.els[1]), op(self.els[2])],
            _phantom: std::marker::PhantomData,
        }
    }

    /// Maps the operation pairwise across two vectors
    fn map_pairwise<U: Copy, V: Copy>(
        self,
        rhs: Tuple<U>,
        op: impl Fn(f64, f64) -> f64,
    ) -> Tuple<V> {
        Tuple {
            els: [
                op(self.els[0], rhs.els[0]),
                op(self.els[1], rhs.els[1]),
                op(self.els[2], rhs.els[2]),
            ],
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Copy, U: Copy> PartialEq<Tuple<U>> for Tuple<T> {
    /// Compare two vectors for equality by comparing the elements using an
    /// epsilon comparison
    fn eq(&self, other: &Tuple<U>) -> bool {
        compare_float(self.els[0], other.els[0])
            && compare_float(self.els[1], other.els[1])
            && compare_float(self.els[2], other.els[2])
    }
}

impl<T: Copy, U> From<U> for Tuple<T>
where
    [f64; 4]: From<U>,
{
    fn from(value: U) -> Self {
        let v: [f64; 4] = value.into();
        Self {
            els: [v[0], v[1], v[2]],
            _phantom: std::marker::PhantomData,
        }
    }
}

// Operations that apply to [Vector]s, but not [Point]s.

impl Vector {
    /// Sum all elements of a vector to a single number
    pub fn sum(self) -> f64 {
        self.els.iter().sum()
    }

    /// Compute the dot product of two vectors
    pub fn dot_product(self, rhs: Self) -> f64 {
        (self * rhs).sum()
    }

    // Note: the dot function is for convenience to maintain capability with
    // The Ray Tracer Challenge book's naming.

    /// Compute the dot product of two vectors
    pub fn dot(self, rhs: Self) -> f64 {
        self.dot_product(rhs)
    }

    /// Compute the cross product of two vectors
    pub fn cross_product(self, rhs: Self) -> Self {
        let u = self;
        let v = rhs;
        Self {
            els: [
                u.els[1] * v.els[2] - v.els[1] * u.els[2],
                v.els[0] * u.els[2] - u.els[0] * v.els[2],
                u.els[0] * v.els[1] - v.els[0] * u.els[1],
            ],
            _phantom: std::marker::PhantomData,
        }
    }

    // Note: the cross function is for convenience to maintain capability with
    // The Ray Tracer Challenge book's naming.

    /// Compute the cross product of two vectors
    pub fn cross(self, rhs: Self) -> Self {
        self.cross_product(rhs)
    }

    /// Compute the norm, that is the square of the dot product, of a vector
    pub fn norm(self) -> f64 {
        self.dot(self).sqrt()
    }

    /// Compute the magnitude of a vector, which is equivalent to the norm of the vector
    pub fn magnitude(self) -> f64 {
        self.norm()
    }

    /// Compute the square of the norm or magnitude of a vector
    pub fn norm_squared(self) -> f64 {
        self.dot(self)
    }

    /// Normalize a vector by dividing it by its norm or magnitude
    pub fn normalize(self) -> Self {
        self / self.norm()
    }

    /// Calculates the reflection of the vector across the normal vector
    pub fn reflect(self, normal: Self) -> Self {
        self - normal * 2.0 * self.dot(normal)
    }
}

// One may add an f64 to any kind of Tuple,
// a Point to a Vector,
// or a Vector to a Vector

impl<T: Copy> Add<f64> for Tuple<T> {
    type Output = Self;

    fn add(self, rhs: f64) -> Self {
        self.map_elementwise(|component| component + rhs)
    }
}

impl Add<Vector> for Point {
    type Output = Self;

    fn add(self, rhs: Vector) -> Self {
        self.map_pairwise(rhs, |a, b| a + b)
    }
}

impl Add for Vector {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        self.map_pairwise(rhs, |a, b| a + b)
    }
}

// One may subtract an f64 from any kind of Tuple,
// or a point from a point, producing a vector
// or a vector from a point, producing a point
// or a vector from a vector, producing a vector

impl<T: Copy> Sub<f64> for Tuple<T> {
    type Output = Self;

    /// Subtract a constant from each element of a self
    fn sub(self, a: f64) -> Self {
        self.map_elementwise(|component| component - a)
    }
}

impl Sub<Point> for Point {
    type Output = Vector;

    fn sub(self, rhs: Point) -> Vector {
        self.map_pairwise(rhs, |a, b| a - b)
    }
}

impl Sub<Vector> for Point {
    type Output = Point;

    fn sub(self, rhs: Vector) -> Point {
        self.map_pairwise(rhs, |a, b| a - b)
    }
}

impl Sub<Vector> for Vector {
    type Output = Vector;

    fn sub(self, rhs: Vector) -> Vector {
        self.map_pairwise(rhs, |a, b| a - b)
    }
}

// One may multiple a Tuple by an f64
// or by the same kind of Tuple

impl<T: Copy> Mul<f64> for Tuple<T> {
    type Output = Self;

    /// Multiply each element of a self by a constant
    fn mul(self, a: f64) -> Self {
        self.map_elementwise(|component| component * a)
    }
}

impl<T: Copy> Mul<Self> for Tuple<T> {
    type Output = Self;

    /// Multiply two selfs, multiplying each element in one self by the corresponding
    /// positional element in the rhs self
    fn mul(self, rhs: Self) -> Self {
        self.map_pairwise(rhs, |a, b| a * b)
    }
}

// One may divide a Tuple by f64

impl<T: Copy> Div<f64> for Tuple<T> {
    type Output = Self;

    /// Divide each element of the self by a constant
    fn div(self, a: f64) -> Self {
        self.map_elementwise(|component| component / a)
    }
}

// One may negate any kind of tuple

impl<T: Copy> Neg for Tuple<T> {
    type Output = Self;

    /// Negate each element of the self
    fn neg(self) -> Self {
        self.map_elementwise(|component| -component)
    }
}

#[cfg(test)]
mod tests {
    use core::f64;

    use super::*;

    // "A tuple with w=1.0 is a point"
    // "A tuple with w=0 is a vector"
    // These tests are not implemented because points and vectors are two distinct types in
    // this implementation.

    #[test]
    fn point_creates_tuples() {
        let p = point(4.0, -4.0, 3.0);
        let q = Point::new([4.0, -4.0, 3.0]);
        assert_eq!(p, q);
    }

    #[test]
    fn vector_creates_tuples() {
        let u = vector(4.0, -4.0, 3.0);
        let v = Vector::new([4.0, -4.0, 3.0]);
        assert_eq!(u, v);
    }

    #[test]
    fn adding_a_vector_to_a_point() {
        let p = point(3.0, -2.0, 5.0);
        let v = vector(-2.0, 3.0, 1.0);
        assert_eq!(p + v, point(1.0, 1.0, 6.0));
    }

    #[test]
    fn adding_a_vector_to_a_vector() {
        let u = vector(1.0, 2.0, 3.0);
        let v = vector(4.0, 5.5, 6.5);
        assert_eq!(u + v, vector(5.0, 7.5, 9.5));
    }

    #[test]
    fn subtracting_two_points() {
        let p = point(3.0, 2.0, 1.0);
        let q = point(5.0, 6.0, 7.0);
        assert_eq!(p - q, vector(-2.0, -4.0, -6.0));
    }

    #[test]
    fn subtracting_a_vector_from_a_point() {
        let p = point(3.0, 2.0, 1.0);
        let v = vector(5.0, 6.0, 7.0);
        assert_eq!(p - v, point(-2.0, -4.0, -6.0));
    }

    #[test]
    fn subtracting_two_vectors() {
        let u = vector(3.0, 2.0, 1.0);
        let v = vector(5.0, 6.0, 7.0);
        assert_eq!(u - v, vector(-2.0, -4.0, -6.0));
    }

    #[test]
    fn subtracting_a_vector_from_the_zero_vector() {
        let zero = vector(0.0, 0.0, 0.0);
        let v = vector(1.0, -2.0, 3.0);
        assert_eq!(zero - v, vector(-1.0, 2.0, -3.0));
    }

    #[test]
    fn negating_a_vector() {
        let v = vector(1.0, -2.0, 3.0);
        assert_eq!(-v, vector(-1.0, 2.0, -3.0));
    }

    #[test]
    fn multiplying_a_point_by_a_scalar() {
        let p = point(1.0, -2.0, 3.0);
        assert_eq!(p * 3.5, point(3.5, -7.0, 10.5));
    }

    #[test]
    fn multiplying_a_vector_by_a_scalar() {
        let v = vector(1.0, -2.0, 3.0);
        assert_eq!(v * 3.5, vector(3.5, -7.0, 10.5));
    }

    #[test]
    fn multiplying_a_vector_by_a_fraction() {
        let v = vector(1.0, -2.0, 3.0);
        assert_eq!(v * 0.5, vector(0.5, -1.0, 1.5));
    }

    #[test]
    fn dividing_a_point_by_a_scalar() {
        let p = point(1.0, -2.0, 3.0);
        assert_eq!(p / 2.0, point(0.5, -1.0, 1.5));
    }

    #[test]
    fn dividing_a_vector_by_a_scalar() {
        let v = vector(1.0, -2.0, 3.0);
        assert_eq!(v / 2.0, vector(0.5, -1.0, 1.5));
    }

    #[test]
    fn computing_the_magnitude_of_vector_1_0_0() {
        let v = vector(1.0, 0.0, 0.0);
        assert_eq!(v.magnitude(), 1.0);
    }

    #[test]
    fn computing_the_magnitude_of_vector_0_1_0() {
        let v = vector(0.0, 1.0, 0.0);
        assert_eq!(v.magnitude(), 1.0);
    }

    #[test]
    fn computing_the_magnitude_of_vector_0_0_1() {
        let v = vector(0.0, 0.0, 1.0);
        assert_eq!(v.magnitude(), 1.0);
    }

    #[test]
    fn computing_the_magnitude_of_vector_1_2_3() {
        let v = vector(1.0, 2.0, 3.0);
        assert_eq!(v.magnitude(), f64::sqrt(14.0));
    }

    #[test]
    fn computing_the_magnitude_of_vector_neg_1_neg_2_neg_3() {
        let v = vector(-1.0, -2.0, -3.0);
        assert_eq!(v.magnitude(), f64::sqrt(14.0));
    }

    #[test]
    fn normalizing_vector_4_0_0_gives_vector_1_0_0() {
        let v = vector(4.0, 0.0, 0.0);
        assert_eq!(v.normalize(), vector(1.0, 0.0, 0.0));
    }

    #[test]
    fn normalizing_vector_1_2_3() {
        let v = vector(1.0, 2.0, 3.0);
        assert_eq!(
            v.normalize(),
            vector(
                1.0 / f64::sqrt(14.0),
                2.0 / f64::sqrt(14.0),
                3.0 / f64::sqrt(14.0)
            )
        );
    }

    #[test]
    fn the_magnitude_of_a_normalized_vector() {
        let v = vector(1.0, 2.0, 3.0);
        assert_eq!(v.normalize().magnitude(), 1.0);
    }

    #[test]
    fn the_dot_product_of_two_vectors() {
        let u = vector(1.0, 2.0, 3.0);
        let v = vector(2.0, 3.0, 4.0);
        assert_eq!(u.dot(v), 20.0);
    }

    #[test]
    fn the_cross_product_of_two_vectors() {
        let a = vector(1.0, 2.0, 3.0);
        let b = vector(2.0, 3.0, 4.0);
        assert_eq!(a.cross(b), vector(-1.0, 2.0, -1.0));
        assert_eq!(b.cross(a), vector(1.0, -2.0, 1.0));
    }

    #[test]
    fn reflecting_a_vector_approaching_at_45_degrees() {
        let v = vector(1.0, -1.0, 0.0);
        let n = vector(0.0, 1.0, 0.0);
        assert_eq!(v.reflect(n), vector(1.0, 1.0, 0.0));
    }

    #[test]
    fn reflecting_a_vector_off_a_slanted_surface() {
        let v = vector(0.0, -1.0, 0.0);
        let n = vector(f64::consts::SQRT_2 / 2.0, f64::consts::SQRT_2 / 2.0, 0.0);
        assert_eq!(v.reflect(n), vector(1.0, 0.0, 0.0));
    }
}
