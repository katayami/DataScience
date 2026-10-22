import unittest

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput
from python_basics import (
    count_one_bits, count_vowels, has_unique_characters, is_balanced_number,
    mse, multiplicative_persistence, prime_factorization, pyramid,
)


class CountVowelsTest(unittest.TestCase):
    def test_mixed_case(self):
        self.assertEqual(count_vowels(TextInput("HelloWOrld")), 3)

    def test_all_vowels(self):
        self.assertEqual(count_vowels(TextInput("aEiOu")), 5)

    def test_no_vowels(self):
        self.assertEqual(count_vowels(TextInput("rhythm")), 0)

    def test_empty(self):
        self.assertEqual(count_vowels(TextInput("")), 0)


class HasUniqueCharactersTest(unittest.TestCase):
    def test_unique(self):
        self.assertTrue(has_unique_characters(TextInput("abcdef")))

    def test_repeated(self):
        self.assertFalse(has_unique_characters(TextInput("hello")))

    def test_case_sensitive(self):
        self.assertTrue(has_unique_characters(TextInput("aA")))

    def test_empty_and_single(self):
        self.assertTrue(has_unique_characters(TextInput("")))
        self.assertTrue(has_unique_characters(TextInput("x")))


class CountOneBitsTest(unittest.TestCase):
    def test_values(self):
        self.assertEqual(count_one_bits(PositiveIntegerInput(1)), 1)
        self.assertEqual(count_one_bits(PositiveIntegerInput(7)), 3)
        self.assertEqual(count_one_bits(PositiveIntegerInput(8)), 1)
        self.assertEqual(count_one_bits(PositiveIntegerInput(255)), 8)

    def test_large(self):
        self.assertEqual(count_one_bits(PositiveIntegerInput(2**100 - 1)), 100)


class MultiplicativePersistenceTest(unittest.TestCase):
    def test_examples(self):
        self.assertEqual(multiplicative_persistence(PositiveIntegerInput(39)), 3)
        self.assertEqual(multiplicative_persistence(PositiveIntegerInput(4)), 0)
        self.assertEqual(multiplicative_persistence(PositiveIntegerInput(999)), 4)

    def test_edge_cases(self):
        self.assertEqual(multiplicative_persistence(PositiveIntegerInput(10)), 1)
        self.assertEqual(multiplicative_persistence(PositiveIntegerInput(9)), 0)
        self.assertEqual(multiplicative_persistence(PositiveIntegerInput(277777788888899)), 11)


class MseTest(unittest.TestCase):
    def test_basic(self):
        self.assertAlmostEqual(mse(VectorPairInput([1, 2, 3], [1, 2, 5])), 4 / 3)

    def test_equal_vectors(self):
        self.assertEqual(mse(VectorPairInput([5, -3], [5, -3])), 0.0)

    def test_single_element(self):
        self.assertEqual(mse(VectorPairInput([-2], [3])), 25.0)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            mse(VectorPairInput([1, 2], [1]))
        with self.assertRaises(ValueError):
            mse(VectorPairInput([], []))


class PrimeFactorizationTest(unittest.TestCase):
    def test_example(self):
        self.assertEqual(prime_factorization(PositiveIntegerInput(86240)), "(2**5)(5)(7**2)(11)")

    def test_prime(self):
        self.assertEqual(prime_factorization(PositiveIntegerInput(2)), "(2)")
        self.assertEqual(prime_factorization(PositiveIntegerInput(97)), "(97)")

    def test_powers(self):
        self.assertEqual(prime_factorization(PositiveIntegerInput(1024)), "(2**10)")
        self.assertEqual(prime_factorization(PositiveIntegerInput(7775460)), "(2**2)(3**3)(5)(7)(11**2)(17)")

    def test_one(self):
        self.assertEqual(prime_factorization(PositiveIntegerInput(1)), "")


class PyramidTest(unittest.TestCase):
    def test_possible(self):
        self.assertEqual(pyramid(PositiveIntegerInput(1)), 1)
        self.assertEqual(pyramid(PositiveIntegerInput(5)), 2)
        self.assertEqual(pyramid(PositiveIntegerInput(14)), 3)
        self.assertEqual(pyramid(PositiveIntegerInput(338350)), 100)

    def test_impossible(self):
        self.assertEqual(pyramid(PositiveIntegerInput(2)), "It is impossible")
        self.assertEqual(pyramid(PositiveIntegerInput(15)), "It is impossible")


class IsBalancedNumberTest(unittest.TestCase):
    def test_examples(self):
        self.assertTrue(is_balanced_number(PositiveIntegerInput(1234006)))
        self.assertFalse(is_balanced_number(PositiveIntegerInput(123456)))

    def test_short_numbers(self):
        self.assertTrue(is_balanced_number(PositiveIntegerInput(7)))
        self.assertTrue(is_balanced_number(PositiveIntegerInput(12)))
        self.assertTrue(is_balanced_number(PositiveIntegerInput(959)))
        self.assertFalse(is_balanced_number(PositiveIntegerInput(1230)))

    def test_even_length(self):
        self.assertTrue(is_balanced_number(PositiveIntegerInput(295592)))
        self.assertTrue(is_balanced_number(PositiveIntegerInput(1001)))


if __name__ == "__main__":
    unittest.main()
