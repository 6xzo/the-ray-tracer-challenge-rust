use std::ops::{Index, Mul};

use crate::{
    tuple::Tuple,
    utilities::{compare_float, from_fn_2d},
};

/// Represents special matrix types
pub enum MatrixType {
    /// A matrix where every element is the same constant
    Constant(f64),
    /// A matrix where every element is zero except the diagonal, which
    /// has the same constant value
    Diagonal(f64),
    /// A square matrix with ones on the diagonal and zeros elsewhere
    Identity,
}

/// An immutable ROWS x COLUMNS matrix of f64 values
#[derive(Debug, Copy, Clone)]
pub struct Matrix<const ROWS: usize, const COLUMNS: usize> {
    data: [[f64; COLUMNS]; ROWS],
}

impl<const ROWS: usize, const COLUMNS: usize> Matrix<ROWS, COLUMNS> {
    /// Creates a new matrix with the given 2D array data
    pub fn new(data: [[f64; COLUMNS]; ROWS]) -> Self {
        Self { data }
    }

    /// Creates a new matrix of the given special type. This is intended to make it
    /// easier to create matrix types that would have a lot of repeated entries for
    /// zeroes and other constants.
    pub fn new_by_type(matrix_type: MatrixType) -> Self {
        match matrix_type {
            MatrixType::Constant(value) => Self {
                data: [[value; COLUMNS]; ROWS],
            },

            MatrixType::Diagonal(value) => {
                let data = from_fn_2d(|row, column| if row == column { value } else { 0.0 });
                Self { data }
            }

            MatrixType::Identity => {
                let data = from_fn_2d(|row, column| if row == column { 1.0 } else { 0.0 });
                Self { data }
            }
        }
    }

    /// Transposes the matrix and returns the transpose as a new matrix
    pub fn transpose(&self) -> Matrix<COLUMNS, ROWS> {
        let mut result = [[0.0; ROWS]; COLUMNS];
        for (row_index, row) in self.data.iter().enumerate() {
            for (column_index, &value) in row.iter().enumerate() {
                result[column_index][row_index] = value;
            }
        }
        Matrix::new(result)
    }
}

impl<const ROWS: usize, const COLUMNS: usize> Index<(usize, usize)> for Matrix<ROWS, COLUMNS> {
    type Output = f64;

    /// Implements indexing for `Matrix` so that a matrix, such as `m`, can be indexed
    /// using `m[(row, column)]` to get the value at that row and column.
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.data[index.0][index.1]
    }
}

impl<const ROWS: usize, const COLUMNS: usize> PartialEq for Matrix<ROWS, COLUMNS> {
    /// Compare two matrices for equality by comparing the elements using an epsilon
    /// comparison. I would have preferred doing this a more "functional" way using
    /// a higher-order function, but this is simple and works, as I couldn't find a
    /// pre-included way in the Rust standard library for this.
    fn eq(&self, other: &Matrix<ROWS, COLUMNS>) -> bool {
        // Here's a way to do it with iterators...
        let a = self.data.iter().flat_map(|v| v.iter());
        let b = other.data.iter().flat_map(|v| v.iter());
        a.zip(b).all(|(&a, &b)| compare_float(a, b))
    }
}

impl<const ROWS: usize, const COLUMNS: usize, const COLUMNS2: usize> Mul<Matrix<COLUMNS, COLUMNS2>>
    for Matrix<ROWS, COLUMNS>
{
    type Output = Matrix<ROWS, COLUMNS2>;

    /// Multiplies two matrices. Note that whether the matrices can be multiplied or
    /// not is enforced by the type system.
    fn mul(self, other: Matrix<COLUMNS, COLUMNS2>) -> Matrix<ROWS, COLUMNS2> {
        let mut result = [[0.0; COLUMNS2]; ROWS];
        for row in 0..ROWS {
            for col in 0..COLUMNS2 {
                result[row][col] = (0..COLUMNS).map(|k| self[(row, k)] * other[(k, col)]).sum();
            }
        }
        Matrix::new(result)
    }
}

impl<T: Copy> Mul<Tuple<T>> for Matrix<4, 4> {
    type Output = Tuple<T>;

    /// Multiplies a tuple on the left by a 4x4 matrix and returns a new tuple
    fn mul(self, other: Tuple<T>) -> Tuple<T> {
        let tuple: Matrix<4, 1> = Matrix::new(other.col4());
        let result = self * tuple;
        Tuple::from(result.transpose().data[0])
    }
}

//############################################################
//#### Tests #################################################
//############################################################

#[cfg(test)]
mod tests {
    use crate::tuple::point;

    use super::*;

    // "A tuple with w=1.0 is a point"
    // "A tuple with w=0 is a vector"
    // These tests are not implemented because points and vectors are two distinct types in
    // this implementation.

    #[test]
    fn constructing_and_inspecting_a_4x4_matrix() {
        let m: Matrix<4, 4> = Matrix::new([
            [1.0, 2.0, 3.0, 4.0],
            [5.5, 6.5, 7.5, 8.5],
            [9.0, 10.0, 11.0, 12.0],
            [13.5, 14.5, 15.5, 16.5],
        ]);
        assert_eq!(
            [
                m[(0, 0)],
                m[(0, 3)],
                m[(1, 0)],
                m[(1, 2)],
                m[(2, 2)],
                m[(3, 0)],
                m[(3, 2)]
            ],
            [1.0, 4.0, 5.5, 7.5, 11.0, 13.5, 15.5]
        );
    }

    #[test]
    fn a_2x2_matrix_ought_to_be_representable() {
        let m: Matrix<2, 2> = Matrix::new([[-3.0, 5.0], [1.0, -2.0]]);
        assert_eq!(
            [m[(0, 0)], m[(0, 1)], m[(1, 0)], m[(1, 1)]],
            [-3.0, 5.0, 1.0, -2.0]
        );
    }

    #[test]
    fn a_3x3_matrix_ought_to_be_representable() {
        let m: Matrix<3, 3> = Matrix::new([[-3.0, 5.0, 0.0], [1.0, -2.0, -7.0], [0.0, 1.0, 1.0]]);
        assert_eq!([m[(0, 0)], m[(1, 1)], m[(2, 2)]], [-3.0, -2.0, 1.0]);
    }

    #[test]
    fn matrix_equality_with_identical_matrices() {
        let a: Matrix<4, 4> = Matrix::new([
            [1.000001, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 8.0, 7.0, 6.0],
            [5.0, 4.0, 3.0, 2.0],
        ]);
        let b: Matrix<4, 4> = Matrix::new([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 8.0, 7.0, 6.0],
            [5.0, 4.0, 3.0, 2.0],
        ]);
        assert_eq!(a, b);
    }

    #[test]
    fn matrix_equality_with_different_matrices() {
        let a: Matrix<4, 4> = Matrix::new([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 8.0, 7.0, 6.0],
            [5.0, 4.0, 3.0, 2.0],
        ]);
        let b: Matrix<4, 4> = Matrix::new([
            [2.0, 3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0, 9.0],
            [8.0, 7.0, 6.0, 5.0],
            [4.0, 3.0, 2.0, 1.0],
        ]);
        assert_ne!(a, b);
    }

    #[test]
    fn multiplying_two_matrices() {
        let a: Matrix<4, 4> = Matrix::new([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 8.0, 7.0, 6.0],
            [5.0, 4.0, 3.0, 2.0],
        ]);
        let b: Matrix<4, 4> = Matrix::new([
            [-2.0, 1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0, -1.0],
            [4.0, 3.0, 6.0, 5.0],
            [1.0, 2.0, 7.0, 8.0],
        ]);
        let expected_result: Matrix<4, 4> = Matrix::new([
            [20.0, 22.0, 50.0, 48.0],
            [44.0, 54.0, 114.0, 108.0],
            [40.0, 58.0, 110.0, 102.0],
            [16.0, 26.0, 46.0, 42.0],
        ]);
        assert_eq!(a * b, expected_result);
    }

    #[test]
    fn a_matrix_multiplied_by_a_tuple() {
        let a: Matrix<4, 4> = Matrix::new([
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 4.0, 4.0, 2.0],
            [8.0, 6.0, 4.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);
        let b = point(1.0, 2.0, 3.0);
        assert_eq!(a * b, point(18.0, 24.0, 33.0));
    }

    #[test]
    fn multiplying_a_matrix_by_the_identity_matrix() {
        let a: Matrix<4, 4> = Matrix::new([
            [0.0, 1.0, 2.0, 4.0],
            [1.0, 2.0, 4.0, 8.0],
            [2.0, 4.0, 8.0, 16.0],
            [4.0, 8.0, 16.0, 32.0],
        ]);
        let identity: Matrix<4, 4> = Matrix::new_by_type(MatrixType::Identity);
        assert_eq!(a * identity, a);
    }

    #[test]
    fn multiplying_the_identity_matrix_by_a_tuple() {
        let identity: Matrix<4, 4> = Matrix::new_by_type(MatrixType::Identity);
        let a = point(1.0, 2.0, 3.0);
        assert_eq!(identity * a, a);
    }

    #[test]
    fn transposing_a_matrix() {
        let m: Matrix<4, 4> = Matrix::new([
            [0.0, 9.0, 3.0, 0.0],
            [9.0, 8.0, 0.0, 8.0],
            [1.0, 8.0, 5.0, 3.0],
            [0.0, 0.0, 5.0, 8.0],
        ]);
        let expected_transpose: Matrix<4, 4> = Matrix::new([
            [0.0, 9.0, 1.0, 0.0],
            [9.0, 8.0, 8.0, 0.0],
            [3.0, 0.0, 5.0, 5.0],
            [0.0, 8.0, 3.0, 8.0],
        ]);
        assert_eq!(m.transpose(), expected_transpose);
    }

    #[test]
    fn transposing_the_identity_matrix() {
        // Note that this also notionally tests different construction methods
        let id1: Matrix<4, 4> = Matrix::new_by_type(MatrixType::Identity);
        let id2: Matrix<4, 4> = Matrix::new_by_type(MatrixType::Diagonal(1.0));
        assert_eq!(id1.transpose(), id2);
    }
}
