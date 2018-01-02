#include <stan/math/rev/mat.hpp>
#include <gtest/gtest.h>
#include <vector>

using stan::length;

typedef Eigen::Matrix<stan::math::var, Eigen::Dynamic, Eigen::Dynamic> var_t1;
typedef std::vector<var_t1> var_t2;
typedef std::vector<var_t2> var_t3;

typedef Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1> var_u1;
typedef std::vector<var_u1> var_u2;
typedef std::vector<var_u2> var_u3;

typedef Eigen::Matrix<stan::math::var, 1, Eigen::Dynamic> var_v1;
typedef std::vector<var_v1> var_v2;
typedef std::vector<var_v2> var_v3;

TEST(MetaTraits, containsNonconstantStruct) {
  using stan::contains_nonconstant_struct;
  using std::vector;
  using Eigen::Matrix;
  using Eigen::Dynamic;

  EXPECT_TRUE(contains_nonconstant_struct<var_t1>::value);
  EXPECT_TRUE(contains_nonconstant_struct<var_t2>::value);
  EXPECT_TRUE(contains_nonconstant_struct<var_t3>::value);
  EXPECT_TRUE(contains_nonconstant_struct<var_u1>::value);
  EXPECT_TRUE(contains_nonconstant_struct<var_u2>::value);
  EXPECT_TRUE(contains_nonconstant_struct<var_u3>::value);
  EXPECT_TRUE(contains_nonconstant_struct<var_v1>::value);
  EXPECT_TRUE(contains_nonconstant_struct<var_v2>::value);
  EXPECT_TRUE(contains_nonconstant_struct<var_v3>::value);

  bool temp
      = contains_nonconstant_struct<var_v3, var_v2, var_v1, double, int>::value;
  EXPECT_TRUE(temp);
}
