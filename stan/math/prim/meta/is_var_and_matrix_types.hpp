#ifndef STAN_MATH_PRIM_META_IS_VAR_AND_MATRIX_TYPES_HPP
#define STAN_MATH_PRIM_META_IS_VAR_AND_MATRIX_TYPES_HPP

#include <stan/math/prim/meta/is_var.hpp>
#include <stan/math/prim/meta/is_matrix.hpp>

namespace stan {

/** \ingroup type_trait
 * Extends std::true_type when instantiated with one type that has a var `scalar_type` and
 * another that is a matrix. Extends std::false_type otherwise.
 * @tparam Types Types to test
 */
template <typename... Types>
using is_var_and_matrix_types = bool_constant<is_var<return_type_t<Types...>>::value &&
  stan::math::disjunction<is_matrix<Types>...>::value>;

template <typename... Types>
using require_all_var_and_matrix_types = require_t<is_var_and_matrix_types<Types...>>;

template <typename... Types>
using require_all_not_var_and_matrix_types = require_not_t<is_var_and_matrix_types<Types...>>;


}  // namespace stan
#endif
