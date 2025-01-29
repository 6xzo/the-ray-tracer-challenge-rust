use std::cmp::PartialEq;
use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::utilities::compare_float;

// Tuples are intentionally separated out into separate types of Vector and Point.
// This is because they are distinctly two different types with enough variation
// that they don't belong as a single type. The ITuple interface provides shared
// behavior, that primarily allows Vector and Point to be transformed with matrices.
// There is a small hit in duplicated code, but this duplication is small and contained
// whereas the correct type design pays off in the overall design of the ray tracer.

/// Trait used to convert 3D tuple-like elements to and from a 4D tuple array
pub trait Tuple {
    type TupleType;

    /// The first element of the tuple-like element
    fn x1(&self) -> f64;

    /// The second element of the tuple-like element
    fn x2(&self) -> f64;

    /// The third element of the tuple-like element
    fn x3(&self) -> f64;

    /// Converts a 3D tuple-like element to an array of length four consisting of
    /// the three components plus a fourth component. The array is essentially the
    /// homogeneous coordinate representation of the tuple.
    fn to_tuple_array(&self) -> [f64; 4];

    /// Creates a 3D tuple-like element from a tuple array
    fn from_tuple_array(t: [f64; 3]) -> Self::TupleType;
}

//############################################################
//#### Vector ################################################
//############################################################

#[derive(Debug, Copy, Clone)]
/// Represents a 3D vector
pub struct Vector {
    i: f64,
    j: f64,
    k: f64,
}

impl Vector {
    /// Maps the operation to each element of the vector
    fn map_elementwise(op: impl Fn(f64) -> f64, v: Vector) -> Vector {
        Vector {
            i: op(v.i),
            j: op(v.j),
            k: op(v.k),
        }
    }

    /// Maps the operation pairwise across two vectors
    fn map_pairwise(op: impl Fn(f64, f64) -> f64, u: Vector, v: Vector) -> Vector {
        Vector {
            i: op(u.i, v.i),
            j: op(u.j, v.j),
            k: op(u.k, v.k),
        }
    }
}

impl Add<f64> for Vector {
    type Output = Vector;

    /// Add a constant to each element of a vector
    fn add(self, a: f64) -> Vector {
        Vector::map_elementwise(|component| component + a, self)
    }
}

impl Add<Vector> for Vector {
    type Output = Vector;

    /// Add two vectors
    fn add(self, other: Vector) -> Vector {
        Vector::map_pairwise(|a, b| a + b, self, other)
    }
}

impl Sub<f64> for Vector {
    type Output = Vector;

    /// Subtract a constant from each element of a vector
    fn sub(self, a: f64) -> Vector {
        Vector::map_elementwise(|component| component - a, self)
    }
}

impl Sub<Vector> for Vector {
    type Output = Vector;

    /// Subtract two vectors
    fn sub(self, other: Vector) -> Vector {
        Vector::map_pairwise(|a, b| a - b, self, other)
    }
}

impl Mul<f64> for Vector {
    type Output = Vector;

    /// Multiply each element of a vector by a constant
    fn mul(self, a: f64) -> Vector {
        Vector::map_elementwise(|component| component * a, self)
    }
}

impl Mul<Vector> for Vector {
    type Output = Vector;

    /// Multiply two vectors, multiplying each element in one vector by the corresponding
    /// positional element in the other vector
    fn mul(self, other: Vector) -> Vector {
        Vector::map_pairwise(|a, b| a * b, self, other)
    }
}

impl Div<f64> for Vector {
    type Output = Vector;

    /// Divide each element of the vector by a constant
    fn div(self, a: f64) -> Vector {
        Vector::map_elementwise(|component| component / a, self)
    }
}

impl Neg for Vector {
    type Output = Vector;

    /// Negate each element of the vector
    fn neg(self) -> Vector {
        Vector::map_elementwise(|component| -component, self)
    }
}

impl PartialEq for Vector {
    /// Compare two vectors for equality by comparing the elements using an epsilon
    /// comparison
    fn eq(&self, other: &Vector) -> bool {
        compare_float(self.i, other.i)
            && compare_float(self.j, other.j)
            && compare_float(self.k, other.k)
    }
}

impl Tuple for Vector {
    type TupleType = Vector;

    fn x1(&self) -> f64 {
        self.i
    }

    fn x2(&self) -> f64 {
        self.j
    }

    fn x3(&self) -> f64 {
        self.k
    }

    fn to_tuple_array(&self) -> [f64; 4] {
        [self.i, self.j, self.k, 0.0]
    }

    fn from_tuple_array(t: [f64; 3]) -> Vector {
        Vector {
            i: t[0],
            j: t[1],
            k: t[2],
        }
    }
}

pub fn vector(i: f64, j: f64, k: f64) -> Vector {
    Vector { i, j, k }
}

//############################################################
//#### Point #################################################
//############################################################

#[derive(Debug, Copy, Clone)]
/// Represents a 3D point
pub struct Point {
    x: f64,
    y: f64,
    z: f64,
}

impl Point {
    /// Maps the operation to each element of the point
    fn map_elementwise(op: impl Fn(f64) -> f64, p: Point) -> Point {
        Point {
            x: op(p.x),
            y: op(p.y),
            z: op(p.z),
        }
    }

    /// Maps the operation pairwise across two points
    fn map_pairwise(op: impl Fn(f64, f64) -> f64, p: Point, q: Point) -> Point {
        Point {
            x: op(p.x, q.x),
            y: op(p.y, q.y),
            z: op(p.z, q.z),
        }
    }
}

impl Add<f64> for Point {
    type Output = Point;

    /// Add a constant to each element of a point
    fn add(self, a: f64) -> Point {
        Point::map_elementwise(|component| component + a, self)
    }
}

impl Add<Vector> for Point {
    type Output = Point;

    /// Add a vector to a point
    fn add(self, other: Vector) -> Point {
        Point {
            x: self.x + other.i,
            y: self.y + other.j,
            z: self.z + other.k,
        }
    }
}

impl Sub<f64> for Point {
    type Output = Point;

    /// Subtract a constant from each element of a point
    fn sub(self, a: f64) -> Point {
        Point::map_elementwise(|component| component - a, self)
    }
}

impl Sub<Vector> for Point {
    type Output = Point;

    /// Subtract a vector from a point
    fn sub(self, other: Vector) -> Point {
        Point {
            x: self.x - other.i,
            y: self.y - other.j,
            z: self.z - other.k,
        }
    }
}

impl Sub<Point> for Point {
    type Output = Vector;

    /// Subtracts two points, p - q (i.e., self - other) to get a vector that
    /// points from q (other) to p (self)
    fn sub(self, other: Point) -> Vector {
        Vector {
            i: self.x - other.x,
            j: self.y - other.y,
            k: self.z - other.z,
        }
    }
}

impl Mul<f64> for Point {
    type Output = Point;

    /// Multiply each element of a point by a constant
    fn mul(self, a: f64) -> Point {
        Point::map_elementwise(|component| component * a, self)
    }
}

impl Mul<Point> for Point {
    type Output = Point;

    /// Multiply two points, multiplying each element in one point by the corresponding
    /// positional element in the other point
    fn mul(self, other: Point) -> Point {
        Point::map_pairwise(|a, b| a * b, self, other)
    }
}

impl Div<f64> for Point {
    type Output = Point;

    /// Divide each element of the point by a constant
    fn div(self, a: f64) -> Point {
        Point::map_elementwise(|component| component / a, self)
    }
}

impl Neg for Point {
    type Output = Point;

    /// Negate each element of the point
    fn neg(self) -> Point {
        Point::map_elementwise(|component| -component, self)
    }
}

impl PartialEq for Point {
    /// Compare two points for equality by comparing the elements using an epsilon
    /// comparison
    fn eq(&self, other: &Point) -> bool {
        compare_float(self.x, other.x)
            && compare_float(self.y, other.y)
            && compare_float(self.z, other.z)
    }
}

impl Tuple for Point {
    type TupleType = Point;

    fn x1(&self) -> f64 {
        self.x
    }

    fn x2(&self) -> f64 {
        self.y
    }

    fn x3(&self) -> f64 {
        self.z
    }

    fn to_tuple_array(&self) -> [f64; 4] {
        [self.x, self.y, self.z, 1.0]
    }

    fn from_tuple_array(t: [f64; 3]) -> Point {
        Point {
            x: t[0],
            y: t[1],
            z: t[2],
        }
    }
}

pub fn point(x: f64, y: f64, z: f64) -> Point {
    Point { x, y, z }
}

//############################################################
//#### Functions #############################################
//############################################################

/// Sum all elements of a vector to a single number
pub fn sum_vector(v: Vector) -> f64 {
    v.i + v.j + v.k
}

/// Compute the dot product of two vectors
pub fn dot_product(u: Vector, v: Vector) -> f64 {
    sum_vector(u * v)
}

/// Compute the dot product of two vectors
pub fn dot(u: Vector, v: Vector) -> f64 {
    dot_product(u, v)
}

// Note: the dot function is for convenience to maintain capability with
// The Ray Tracer Challenge book's naming.

/// Compute the cross product of two vectors
pub fn cross_product(u: Vector, v: Vector) -> Vector {
    Vector {
        i: u.j * v.k - v.j * u.k,
        j: v.i * u.k - u.i * v.k,
        k: u.i * v.j - v.i * u.j,
    }
}

/// Compute the cross product of two vectors
pub fn cross(u: Vector, v: Vector) -> Vector {
    cross_product(u, v)
}

// Note: the cross function is for convenience to maintain capability with
// The Ray Tracer Challenge book's naming.

/// Compute the norm, that is the square of the dot product, of a vector
pub fn norm(v: Vector) -> f64 {
    dot(v, v).sqrt()
}

/// Compute the magnitude of a vector, which is equivalent to the norma of the vector
pub fn magnitude(v: Vector) -> f64 {
    norm(v)
}

/// Compute the square of the norm or magnitude of a vector
pub fn norm_squared(v: Vector) -> f64 {
    dot(v, v)
}

/// Normalize a vector by dividing it by its norm or magnitude
pub fn normalize(v: Vector) -> Vector {
    v / norm(v)
}

/// Calculates the reflection of the vector across the normal vector
pub fn reflect(vector: Vector, normal: Vector) -> Vector {
    vector - normal * 2.0 * dot(vector, normal)
}

//############################################################
//#### Tests #################################################
//############################################################

#[cfg(test)]
mod tests {
    use super::*;

    // "A tuple with w=1.0 is a point"
    // "A tuple with w=0 is a vector"
    // These tests are not implemented because points and vectors are two distinct types in
    // this implementation.

    #[test]
    fn point_creates_tuples() {
        let p = point(4.0, -4.0, 3.0);
        let q = Point {
            x: 4.0,
            y: -4.0,
            z: 3.0,
        };
        assert_eq!(p, q);
    }

    #[test]
    fn vector_creates_tuples() {
        let u = vector(4.0, -4.0, 3.0);
        let v = Vector {
            i: 4.0,
            j: -4.0,
            k: 3.0,
        };
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
        assert_eq!(magnitude(v), 1.0);
    }

    #[test]
    fn computing_the_magnitude_of_vector_0_1_0() {
        let v = vector(0.0, 1.0, 0.0);
        assert_eq!(magnitude(v), 1.0);
    }

    #[test]
    fn computing_the_magnitude_of_vector_0_0_1() {
        let v = vector(0.0, 0.0, 1.0);
        assert_eq!(magnitude(v), 1.0);
    }

    #[test]
    fn computing_the_magnitude_of_vector_1_2_3() {
        let v = vector(1.0, 2.0, 3.0);
        assert_eq!(magnitude(v), f64::sqrt(14.0));
    }

    #[test]
    fn computing_the_magnitude_of_vector_neg_1_neg_2_neg_3() {
        let v = vector(-1.0, -2.0, -3.0);
        assert_eq!(magnitude(v), f64::sqrt(14.0));
    }

    #[test]
    fn normalizing_vector_4_0_0_gives_vector_1_0_0() {
        let v = vector(4.0, 0.0, 0.0);
        assert_eq!(normalize(v), vector(1.0, 0.0, 0.0));
    }

    #[test]
    fn normalizing_vector_1_2_3() {
        let v = vector(1.0, 2.0, 3.0);
        assert_eq!(
            normalize(v),
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
        assert_eq!(magnitude(normalize(v)), 1.0);
    }

    #[test]
    fn the_dot_product_of_two_vectors() {
        let u = vector(1.0, 2.0, 3.0);
        let v = vector(2.0, 3.0, 4.0);
        assert_eq!(dot(u, v), 20.0);
    }

    #[test]
    fn the_cross_product_of_two_vectors() {
        let a = vector(1.0, 2.0, 3.0);
        let b = vector(2.0, 3.0, 4.0);
        assert_eq!(cross(a, b), vector(-1.0, 2.0, -1.0));
        assert_eq!(cross(b, a), vector(1.0, -2.0, 1.0));
    }

    #[test]
    fn reflecting_a_vector_approaching_at_45_degrees() {
        let v = vector(1.0, -1.0, 0.0);
        let n = vector(0.0, 1.0, 0.0);
        assert_eq!(reflect(v, n), vector(1.0, 1.0, 0.0));
    }

    #[test]
    fn reflecting_a_vector_off_a_slanted_surface() {
        let v = vector(0.0, -1.0, 0.0);
        let n = vector(f64::sqrt(2.0) / 2.0, f64::sqrt(2.0) / 2.0, 0.0);
        assert_eq!(reflect(v, n), vector(1.0, 0.0, 0.0));
    }
}
