
///////////////////////////////////////////////////////////////////////////////
// gfxalgebra.hpp
//
// Linear algebra for graphics.
//
// This header defines a gfx::vector class representing a linear algebra
// vector, and a gfx::matrix class representing a linear algebra matrix.
//
// This file builds upon gfxnumeric.hpp, so you may want to familiarize
// yourself with that header before diving into this one.
//
// Students: all of your work should go in this file, and the only files that
// you need to modify in project 2 are this file, and README.md.
//
///////////////////////////////////////////////////////////////////////////////


#pragma once

#include <algorithm>
#include <array>
#include <cmath>

#include <functional>
#include <numeric>

#include "gfxnumeric.hpp"

namespace gfx {

// Forward-declare the matrix type, because a few vector operations involve
// matrices.
template <typename scalar_type,
          size_t HEIGHT,
          size_t WIDTH>
class matrix;

// A mathematical vector, in the spirit of linear algebra.
//
// This is very different from a general-purpose self-resizing array data
// structure such as std::vector. gfx::vector has a fixed dimension (size)
// and only supports mathematical operations.
//
// scalar_type is the type of each element, which must be a numeric type
// such as double, float, or int. At a minimum, it must be possible to
// assign a scalar_type to 0.
//
// DIMENSION is the size of the vector. Dimension should be positive;
// zero-dimension vectors are technically supported, but seem pointless.
template <typename scalar_type,
          size_t DIMENSION>
class vector {
public:

  // Type aliases.
  using same_type = gfx::vector<scalar_type, DIMENSION>;
  using storage_type = std::array<scalar_type, DIMENSION>;
  using iterator = typename storage_type::iterator;
  using const_iterator = typename storage_type::const_iterator;

private:

  storage_type elements_;

public:

  ////////
  // Constructors and destructor.
  ////////

  // Default constructor. Every element is initialized to zero.
  constexpr vector() noexcept
  : vector(0) { }

  // Copy and move contructors.
  constexpr vector(const same_type&) noexcept = default;
  constexpr vector(same_type&&) noexcept = default;

  // Fill constructor. Every element is initialized to default_value.
  constexpr vector(scalar_type default_value) noexcept { fill(default_value); }

  // Iterator constructor. If the iterator range has fewer than DIMENSION
  // elements, the unspecified elements default to 0. If the iterator range
  // has extra elements, the extras are ignored.
  template <typename input_iterator>
  constexpr vector(input_iterator first, input_iterator last) noexcept {
    auto iter = first;
    for (size_t i = 0; i < DIMENSION; ++i) {
      elements_[i] = (iter == last) ? scalar_type(0) : *iter++;
    }
  }

  // Initializer list constructor. If the list has fewer than DIMENSION
  // elements, the unspecified elements default to 0. If the list
  // has extra elements, the extras are ignored.
  constexpr vector(std::initializer_list<scalar_type> il) noexcept
  : vector(il.begin(), il.end()) { }

  // Destructor.
  ~vector() = default;

  ////////
  // Operator overloads.
  ////////

  constexpr same_type& operator= (const same_type&) noexcept = default;

  constexpr bool operator== (const same_type& rhs) const noexcept {
    return std::equal(elements_.begin(), elements_.end(), rhs.elements_.begin());
  }

  constexpr bool operator!= (const same_type& rhs) const noexcept {
    return !(*this == rhs);
  }

  constexpr const scalar_type& operator[](size_t i) const noexcept {
    assert(is_index(i));
    return elements_[i];
  }

  constexpr scalar_type& operator[](size_t i) noexcept {
    assert(is_index(i));
    return elements_[i];
  }

  constexpr same_type operator+(const same_type& rhs) const noexcept {

    same_type sum_;
    for(size_t i = 0; i < DIMENSION; ++i){
      sum_[i] = elements_[i] + rhs.elements_[i];
    }
    return sum_;
  }

  constexpr same_type operator-() const noexcept {
    same_type negation_;
    for(size_t i = 0; i < DIMENSION; ++i){
      negation_[i] = elements_[i] * -1;
    }
    return negation_;
  }

  constexpr same_type operator-(const same_type& rhs) const noexcept {
    same_type difference_;
    for(size_t i = 0; i < DIMENSION; ++i){
      difference_[i] = elements_[i] - rhs.elements_[i];
    }
    return difference_;
  }

  // Vector-scalar product.
  constexpr same_type operator*(scalar_type rhs) const noexcept {
    same_type product_;
    for(size_t i = 0; i < DIMENSION; ++i){
      product_[i]=elements_[i]*rhs;
    }
    return product_;
  }

  // Vector-vector product (dot product).
  constexpr scalar_type operator*(const same_type& rhs) const noexcept {
    scalar_type result_=  0.0;
    for(size_t i = 0; i < DIMENSION; ++i){
      result_ += (elements_[i]*rhs.elements_[i]);
    }
    return result_;
  }

  // Vector divided by scalar.
  constexpr same_type operator/(scalar_type rhs) const noexcept {
    same_type quotient_;
    for(size_t i = 0; i < DIMENSION; ++i){
      quotient_[i]=elements_[i]/rhs;
    }
    return quotient_;
  }

  // Stream insertion operator, for printing.
  friend std::ostream& operator<<(std::ostream& stream, const same_type& rhs) {
    stream << '<';
    if constexpr (DIMENSION > 0) {
      stream << rhs.elements_[0];
    }
    for (size_t i = 1; i < DIMENSION; ++i) {
      stream << ", " << rhs.elements_[i];
    }
    stream << '>';
    return stream;
  }

  ////////
  // Approximate equality.
  ////////

  // Return true iff each element of this vector is approximately equal to
  // the corresponding element of other, using delta, as determined by
  // gfx::approx_equal.
  constexpr bool approx_equal(const same_type& other, scalar_type delta) const noexcept {
    bool equal = true;
    for(size_t i = 0; i < DIMENSION; ++i){
      if(gfx::approx_equal(elements_[i], other.elements_[i], delta)==false){
        equal = false;
        break;
      }
    }
    return equal;
  }

  ////////
  // Iterators.
  ////////

  constexpr const_iterator begin() const noexcept { return elements_.cbegin(); }
  constexpr const_iterator end  () const noexcept { return elements_.cend  (); }

  constexpr iterator begin() noexcept { return elements_.begin(); }
  constexpr iterator end  () noexcept { return elements_.end  (); }

  ////////
  // Size and indices.
  ////////

  constexpr size_t dimension() const noexcept { return DIMENSION; }

  constexpr bool is_index(size_t i) const noexcept { return (i < DIMENSION); }

  ////////
  // Converting to other types.
  ////////

  // Return a vector of size NEW_DIMENSION, based on this vector.
  // NEW_DIMENSION must be greater than DIMENSION.
  // The first DIMENSION elements are copied from this vector.
  // The remaining, newly-created elements are all initialized to default_value.
  template <size_t NEW_DIMENSION>
  vector<scalar_type, NEW_DIMENSION>
  grow(scalar_type default_value = 0) const noexcept {

    static_assert(NEW_DIMENSION > DIMENSION,
                  "new dimension must be larger than old dimension");

    vector<scalar_type, NEW_DIMENSION> new_vector;
    for(size_t i = 0; i < DIMENSION; ++i){
      new_vector[i] = elements_[i];
    }
    for(size_t i = DIMENSION; i < NEW_DIMENSION; ++i){
      new_vector[i] = default_value;
    }
    return new_vector;
  }

  // Return a vector of size NEW_DIMENSION, based on this vector.
  // NEW_DIMENSION must be less than DIMENSION.
  // The returned vector contains the first NEW_DIMENSION elements of this vector.
  template <size_t NEW_DIMENSION>
  vector<scalar_type, NEW_DIMENSION>
  shrink() const noexcept {

    static_assert(NEW_DIMENSION < DIMENSION,
                  "new dimension must be smaller than old dimension");
    vector<scalar_type, NEW_DIMENSION> new_vector;
    for(size_t i = 0; i < NEW_DIMENSION; ++i){
      new_vector[i] = elements_[i];
    }

    return new_vector;
  }

  // Return a vector of size NEW_DIMENSION, based on this vector.
  // Copies NEW_DIMENSION elements, starting at index start.
  // The specified range of indices must all fit within this vector.
  template <size_t NEW_DIMENSION>
  vector<scalar_type, NEW_DIMENSION>
  subvector(size_t start = 0) const noexcept {

    static_assert(NEW_DIMENSION <= DIMENSION,
                  "new dimension cannot be larger than old dimension");
    assert((start + NEW_DIMENSION) <= DIMENSION);

    vector<scalar_type, NEW_DIMENSION> new_vector;

    size_t trav = 0;
    for(size_t i = start; i < start + NEW_DIMENSION; ++i){
      new_vector[trav] = elements_[i];
      ++trav;
    }

    return new_vector;
  }

  // Convert this vector to a column matrix, i.e. a matrix of height
  // DIMENSION and width 1.
  // The definition of this function needs to come after gfx::matrix, and is
  // near the bottom of this source file.
  constexpr matrix<scalar_type, DIMENSION, 1> to_column_matrix() const noexcept;

  // Convert this vector to a row matrix, i.e. a matrix of height 1
  // and width DIMENSION.
  // The definition of this function needs to come after gfx::matrix, and is
  // near the bottom of this source file.
  constexpr matrix<scalar_type, 1, DIMENSION> to_row_matrix() const noexcept;

  ////////
  // Miscellaneous operations.
  ////////

  // Cross product. This function is only defined on vectors of DIMENSION 3.
  constexpr same_type cross(const same_type& rhs) const noexcept {

    static_assert(3 == DIMENSION,
                  "cross product is only defined for 3D vectors");
    vector<scalar_type, DIMENSION> new_vector;

    new_vector.elements_[0] = ((elements_[1]*rhs.elements_[2])-(elements_[2]*rhs.elements_[1]));
    new_vector.elements_[1] = ((elements_[2]*rhs.elements_[0])-(elements_[0]*rhs.elements_[2]));
    new_vector.elements_[2] = ((elements_[0]*rhs.elements_[1])-(elements_[1]*rhs.elements_[0]));

    return new_vector;
  }

  // Fill; assign every element the value x.
  constexpr void fill(scalar_type x) noexcept { elements_.fill(x); }

  // Return true iff this is a unit vector, i.e. the length of this vector is
  // approximately equal to 1, within delta.
  bool is_unit(scalar_type delta) const noexcept {
    if (gfx::approx_equal(magnitude(),static_cast<scalar_type>(1),delta))
      return true;
    else  return false;
  }

  // Return true iff every element of this vector is == 0.
  constexpr bool is_zero() const noexcept {
    using namespace std::placeholders;
    return std::all_of(elements_.begin(), elements_.end(),
                       std::bind(std::equal_to{}, _1, 0));
  }

  // Return the magnitude of this vector, squared (raised to the second power).
  // Computing this quantity does not require a square root, so it is faster
  // than magnitude().
  constexpr scalar_type magnitude_squared() const noexcept {
    scalar_type temp = 0;
    for (size_t i = 0; i < DIMENSION; ++i) {
      temp += pow(elements_[i],2);
    }
    return temp;
  }

  // Return the magnitude (length) of this vector.
  scalar_type magnitude() const noexcept {
    return sqrt(magnitude_squared());
  }

  // Return a version of this vector that has been normalized.
  // In other words, return a vector with magnitude 1.0, and the same direction
  // as this vector.
  same_type normalized() const noexcept {
    same_type result;
    scalar_type mag = magnitude();
    result = *this / mag;
    return result;
  }
};

// Type aliases for 2, 3, and 4-dimensional vectors.
template <typename scalar_type> using vector2 = vector<scalar_type, 2>;
template <typename scalar_type> using vector3 = vector<scalar_type, 3>;
template <typename scalar_type> using vector4 = vector<scalar_type, 4>;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// A mathematical matrix.
//
// Like gfx::vector, this is intended for linear algebra operations, and is not
// a general purpose multidemensional array data structure.
//
// scalar_type is the type of each element, which must be a number type that
// can be initialized to 0.
// HEIGHT is the number of rows in the matrix.
// WIDTH is the number of columns in the matrix.
//
// This class is intended for graphics applications where matrices are rarely
// larger than 4x4.
template <typename scalar_type,
          size_t HEIGHT,
          size_t WIDTH>
class matrix {
public:

  // Type aliases.
  using same_type = matrix<scalar_type, HEIGHT, WIDTH>;
  using row_type = vector<scalar_type, WIDTH>;
  using row_container_type = std::array<row_type, HEIGHT>;
  using row_iterator = typename row_container_type::iterator;
  using const_row_iterator = typename row_container_type::const_iterator;

private:

  // The rows (elements).
  row_container_type rows_;

public:

  ////////
  // Constructors and destructor.
  ////////

  // Default constructor, which initializes each element to 0.
  constexpr matrix() noexcept
  : matrix(scalar_type(0)) {}

  // Copy constructor.
  constexpr matrix(const same_type&) noexcept = default;

  // Move constructor.
  constexpr matrix(same_type&&) noexcept = default;

  // Fill constructor, each element is initialized to default_value.
  constexpr matrix(scalar_type default_value) noexcept { fill(default_value); }

  // Iterator constructor. Elements are filled in row-major order, i.e.
  // the first row left-to-right, then the second row left-to-right, and so on.
  // If the iterator range contains fewer elements than this matrix, the
  // unspecified elements are initialized to 0.
  // If the iterator range contains more elements than this matrix, the extras
  // are ignored.
  template <typename input_iterator>
  constexpr matrix(input_iterator first, input_iterator last) noexcept {
    input_iterator iter = first;
    for (size_t r = 0; r < HEIGHT; ++r) {
      for (size_t c = 0; c < WIDTH; ++c) {
        rows_[r][c] = (iter == last) ? scalar_type(0) : *iter++;
      }
    }
  }
// Initializer list constructor. Elements are filled in row-major order, i.e.
  // the first row left-to-right, then the second row left-to-right, and so on.
  // If the list contains fewer elements than this matrix, the
  // unspecified elements are initialized to 0.
  // If the list contains more elements than this matrix, the extras
  // are ignored.
  constexpr matrix(std::initializer_list<scalar_type> il) noexcept
  : matrix(il.begin(), il.end()) { }

  ////////
  // Operator overloads.
  ////////

  constexpr same_type& operator= (const same_type&) noexcept = default;

  constexpr bool operator== (const same_type& rhs) const noexcept {
    return std::equal(rows_.begin(), rows_.end(), rhs.rows_.begin());
  }

  constexpr bool operator!= (const same_type& rhs) const noexcept {
    return !(*this == rhs);
  }

  constexpr const row_type& operator[] (size_t row) const noexcept {
    assert(is_row(row));
    return rows_[row];
  }

  constexpr row_type& operator[] (size_t row) noexcept {
    assert(is_row(row));
    return rows_[row];
  }

  constexpr same_type operator+ (const same_type& rhs) const noexcept {
    same_type sum_;
    for(size_t r = 0; r < HEIGHT; ++r){
      for(size_t c = 0; c < WIDTH; ++c){
        sum_.rows_[r][c] = rows_[r][c] + rhs.rows_[r][c];
      }
    }
    return sum_;
  }

  // Negation.
  constexpr same_type operator- () const noexcept {
    same_type negation_;
    for(size_t r = 0; r < HEIGHT; ++r){
      for(size_t c = 0; c < WIDTH; ++c){
        negation_.rows_[r][c] = -1*rows_[r][c];
      }
    }
    return negation_;
  }

  // Matrix-matrix subtraction.
  constexpr same_type operator- (const same_type& rhs) const noexcept {
    same_type difference_;
    for(size_t r = 0; r < HEIGHT; ++r){
      for(size_t c = 0; c < WIDTH; ++c){
        difference_.rows_[r][c] = rows_[r][c] - rhs.rows_[r][c];
      }
    }
    return difference_;
  }

  // Matrix-scalar multiplication.
  constexpr same_type operator* (const scalar_type rhs) const noexcept {
    same_type product_;
    for(size_t r = 0; r < HEIGHT; ++r){
      for(size_t c = 0; c < WIDTH; ++c){
        product_.rows_[r][c] = rows_[r][c] * rhs;
      }
    }
    return product_;
  }

  // Matrix-matrix multiplication.
  // The rhs matrix's height must be the same as this matrix' WIDTH.
  // The result vector's dimension is the same as the rhs matrix's width.
  // These dimensions are specified as template parameters, so trying to
  // multiply with invalid dimensions will cause a compile error.
  template <size_t RESULT_WIDTH>
  constexpr
  matrix<scalar_type, HEIGHT, RESULT_WIDTH>
  operator* (const matrix<scalar_type, WIDTH, RESULT_WIDTH>& rhs) const noexcept {
    matrix<scalar_type, HEIGHT, RESULT_WIDTH> result_;

    for (size_t y = 0; y < HEIGHT; ++y) { // rows
      for (size_t x = 0; x < RESULT_WIDTH; ++x) { // result-width
        for(size_t addies = 0; addies < WIDTH; ++addies) { // column
          result_[y][x] += rows_[y][addies] * rhs[addies][x];
        }
      }
    }

    return result_;
  }

  // Division by a scalar.
  constexpr same_type operator/(scalar_type rhs) const noexcept {
    same_type quotient_;
    for(size_t r = 0; r < HEIGHT; ++r){
      for(size_t c = 0; c < WIDTH; ++c){
        quotient_.rows_[r][c] = rows_[r][c] / rhs;
      }
    }
    return quotient_;
  }

  // Stream insertion operator, for printing.
  friend std::ostream& operator<<(std::ostream& stream, const same_type& rhs) {
    for (auto& row : rhs.rows_) {
      stream << '[';
      if constexpr (WIDTH > 0) {
        stream << row[0];
      }
      for (size_t i = 1; i < WIDTH; ++i) {
        stream << ' ' << row[i];
      }
      stream << ']' << std::endl;
    }
    return stream;
  }

  ////////
  // Approximate equality.
  ////////

  // Return true iff each element of this matrix is approximately equal to
  // the corresponding element of other, using delta, as determined by
  // gfx::approx_equal.
  constexpr bool approx_equal(const same_type& other, scalar_type delta) const noexcept {
    bool equal = true;
    for(size_t r = 0; r < HEIGHT; ++r){
      for(size_t c = 0; c < WIDTH; ++c){
        if(gfx::approx_equal(rows_[r][c], other[r][c],delta)==false) equal = false;
      }
    }
    return equal;
  }

  ////////
  // Iterators.
  ////////

  constexpr const_row_iterator begin() const noexcept { return rows_.begin(); }
  constexpr       row_iterator begin()       noexcept { return rows_.begin(); }

  constexpr const_row_iterator end() const noexcept { return rows_.end(); }
  constexpr       row_iterator end()       noexcept { return rows_.end(); }

  ////////
  // Size and indices.
  ////////

  static constexpr size_t height() noexcept { return HEIGHT; }

  constexpr size_t width() const noexcept { return WIDTH; }

  constexpr bool is_column(size_t column) const noexcept { return (column < WIDTH); }

  constexpr bool is_row(size_t row) const noexcept { return (row < HEIGHT); }

  constexpr bool is_row_column(size_t row, size_t column) const noexcept {
    return is_row(row) && is_column(column);
  }

  template <typename OTHER_SCALAR_TYPE,
            size_t OTHER_WIDTH,
            size_t OTHER_HEIGHT>
  constexpr bool
  is_same_size(const matrix<OTHER_SCALAR_TYPE, OTHER_HEIGHT, OTHER_WIDTH>& other)
  const noexcept {
    return (width() == other.width()) && (height() == other.height());
  }

  static constexpr bool is_square() noexcept { return (WIDTH == HEIGHT); }

  // Return true iff every element of this matrix is == 0.
  constexpr bool is_zero() const noexcept {
    return std::all_of(rows_.begin(), rows_.end(),
                       [](auto& row) { return row.is_zero(); });
  }

  ////////
  // Converting to other types.
  ////////

  // Return column number c of this matrix as a width-1 matrix.
  // c must be a valid column number.
  constexpr matrix<scalar_type, HEIGHT, 1> column_matrix(size_t c) const noexcept {

    assert(is_column(c));
    matrix<scalar_type, HEIGHT, 1> result;
    for(size_t r = 0; r < HEIGHT; ++r){
      result[r][0] = rows_[r][c];
    }
    return result;
  }

  // Return column number c of this matrix as a vector.
  // c must be a valid column number.
  constexpr vector<scalar_type, HEIGHT> column_vector(size_t c) const noexcept {

    assert(is_column(c));
    vector<scalar_type, HEIGHT> result;
    for(size_t r=0; r < HEIGHT; ++r){
      result[r] = rows_[r][c];
    }
    return result;
  }

  // Return row number r of this matrix as a height-1 matrix.
  // r must be a valid row number.
  constexpr const gfx::matrix<scalar_type, 1, WIDTH> row_matrix(size_t r) const noexcept {

    assert(is_row(r));
    matrix<scalar_type, 1, WIDTH> result;
    for(size_t c = 0; c < WIDTH; ++c){
      result[0][c] = rows_[r][c];
    }
    return result;
  }

  // Return row number r of this matrix as a vector.
  // r must be a valid row number.
  constexpr const row_type& row_vector(size_t r) const noexcept {
    assert(is_row(r));
    vector<scalar_type, WIDTH> result;
    for(size_t c=0; c < WIDTH; ++c){
      result[c] = rows_[r][c];
    }
    return rows_[r];
  }

  // Return a portion of this matrix as a new matrix object.
  // The dimensions of the sub-matrix are template parameters NEW_HEIGHT
  // and NEW_WIDTH.
  // The location of the sub-matrix is specified by top_row and left_column.
  // The specified range of locations must all fit within this matrix.
  template <size_t NEW_HEIGHT,
            size_t NEW_WIDTH>
  constexpr matrix<scalar_type, NEW_HEIGHT, NEW_WIDTH>
  submatrix(size_t top_row = 0, size_t left_column = 0) const noexcept {

    assert(is_row_column(top_row, left_column));
    assert((top_row + NEW_HEIGHT) <= HEIGHT);
    assert((left_column + NEW_WIDTH) <= WIDTH);
    matrix<scalar_type, NEW_HEIGHT, NEW_WIDTH> result;

    size_t old_row = top_row;
    size_t old_column = left_column;
    for(size_t r = 0; r<NEW_HEIGHT; ++r){
      for(size_t c = 0; c<NEW_WIDTH; ++c){
        result[r][c] = rows_[old_row][old_column];
        ++old_column;
      }
      old_column = left_column;
      ++old_row;
    }
    return result;
  }

  ////////
  // Determinant.
  ////////

  // Return the determinant of this matrix.
  // This function is only defined on square 2x2 or 3x3 matrices.
  scalar_type determinant() const noexcept {

    static_assert(is_square(),
                  "determinant is only defined for square matrices");
    static_assert((WIDTH == 2) || (WIDTH == 3),
	                "determinant only implemented for 2x2 and 3x3 matrices");
    scalar_type det = 0;
    if(WIDTH == 3){
      scalar_type row0 = rows_[0][0]*((rows_[1][1]*rows_[2][2])-(rows_[2][1]*rows_[1][2]));
      scalar_type row1 = rows_[0][1]*((rows_[1][0]*rows_[2][2])-(rows_[2][0]*rows_[1][2]));
      scalar_type row2 = rows_[0][2]*((rows_[1][0]*rows_[2][1])-(rows_[2][0]*rows_[1][1]));
      det = row0 - row1 + row2;
    }
    else{
      det = (rows_[0][0]*rows_[1][1])-(rows_[1][0]*rows_[0][1]);
    }
    return det;
  }

  ////////
  // Solving linear systems.
  ////////

  // Solve a linear system.
  // This matrix is considered to be the M matrix of coefficients.
  // b is the vector containing scalars on the right-hand-side of the equations.
  // Returns the x vector of values for each variable in the system.
  // This function is only defined for square 2x2 or 3x3 matrices.
  vector<scalar_type, HEIGHT> solve(const vector<scalar_type, HEIGHT>& b) const noexcept {

    static_assert(is_square(),
                  "only square linear systems can be solved");
    static_assert((WIDTH == 2) || (WIDTH == 3),
                  "solve is only implemented for 2x2 and 3x3 matrices");
    vector<scalar_type, HEIGHT> result;
    //Cramer's Rule
    scalar_type det = 0;
    if(WIDTH == 3){
      scalar_type row0 = rows_[0][0]*((rows_[1][1]*rows_[2][2])-(rows_[2][1]*rows_[1][2]));
      scalar_type row1 = rows_[0][1]*((rows_[1][0]*rows_[2][2])-(rows_[2][0]*rows_[1][2]));
      scalar_type row2 = rows_[0][2]*((rows_[1][0]*rows_[2][1])-(rows_[2][0]*rows_[1][1]));
      det = row0 - row1 + row2;
    }
    else{
      det = (rows_[0][0]*rows_[1][1])-(rows_[1][0]*rows_[0][1]);
    }

    if(WIDTH ==3){
      scalar_type row0;
      scalar_type row1;
      scalar_type row2;
      //x
      row0 = b[0]*((rows_[1][1]*rows_[2][2])-(rows_[2][1]*rows_[1][2]));
      row1 = rows_[0][1]*((b[1]*rows_[2][2])-(b[2]*rows_[1][2]));
      row2 = rows_[0][2]*((b[1]*rows_[2][1])-(b[2]*rows_[1][1]));
      result[0]= (row0 - row1 + row2)/det;
      //y
      row0 = rows_[0][0]*((b[1]*rows_[2][2])-(b[2]*rows_[1][2]));
      row1 = b[0]*((rows_[1][0]*rows_[2][2])-(rows_[2][0]*rows_[1][2]));
      row2 = rows_[0][2]*((rows_[1][0]*b[2])-(rows_[2][0]*b[1]));
      result[1] = (row0 - row1 + row2)/det;
      //z
      row0 = rows_[0][0]*((rows_[1][1]*b[2])-(rows_[2][1]*b[1]));
      row1 = rows_[0][1]*((rows_[1][0]*b[2])-(rows_[2][0]*b[1]));
      row2 = b[0]*((rows_[1][0]*rows_[2][1])-(rows_[2][0]*rows_[1][1]));
      result[2] = (row0 - row1 + row2)/det;
    }
    else{
      result[0] = ((b[0]*rows_[1][1])-(b[1]*rows_[0][1]))/det;
      result[1] = ((rows_[0][0]*b[1])-(rows_[1][0]*b[0]))/det;
    }
    return result;
  }

  ////////
  // Miscellaneous operations.
  ////////

  // Fill; assign each element to x.
  constexpr void fill(scalar_type x) noexcept {
    std::for_each(rows_.begin(), rows_.end(),
                  [&](auto& row) { row.fill(x); });
  }

  // Create and return an identity matrix with the same dimensions as this
  // matrix.
  // This function is only defined for square matrices.
  static constexpr same_type identity() noexcept {
    same_type ident;
    for(size_t i = 0; i < WIDTH && i < HEIGHT; ++i) {
      ident[i][i] = 1;
    }
    return ident;
  }

  // Return the transpose of this matrix, i.e. exchange rows with columns.
  constexpr matrix<scalar_type, WIDTH, HEIGHT> transpose() const noexcept {
    matrix<scalar_type, HEIGHT, WIDTH> result;
    for(size_t r = 0; r < HEIGHT; ++r){
      for(size_t c = 0; c < WIDTH; ++c){
        result[c][r] = rows_[r][c];
      }
    }
    return result;
  }
};

// Type aliases for 2x2, 3x3, and 4x4 square matrices.
template <typename scalar_type> using matrix2x2 = matrix<scalar_type, 2, 2>;
template <typename scalar_type> using matrix3x3 = matrix<scalar_type, 3, 3>;
template <typename scalar_type> using matrix4x4 = matrix<scalar_type, 4, 4>;

// Now we can finally define vector::to_column_matrix and vector::to_row_matrix.

template <typename scalar_type,
          size_t DIMENSION>
constexpr matrix<scalar_type, DIMENSION, 1>
vector<scalar_type, DIMENSION>::to_column_matrix() const noexcept {
  matrix<scalar_type, DIMENSION, 1> mat;
  for (size_t i = 0; i < DIMENSION; ++i)
    mat[i][0] = elements_[i];
  return mat;
}

template <typename scalar_type,
          size_t DIMENSION>
constexpr matrix<scalar_type, 1, DIMENSION>
vector<scalar_type, DIMENSION>::to_row_matrix() const noexcept {
  matrix<scalar_type, 1, DIMENSION> mat;
  for (size_t i = 0; i < DIMENSION; ++i)
    mat[0][i] = elements_[i];
  return mat;
}

} // namespace gfx
