import unittest
import torch
from twin4build.utils.types import Scalar, Vector, Parameter


class TestScalar(unittest.TestCase):
    def test_scalar_initialization(self):
        """Test scalar initialization with and without initial value."""
        s = Scalar()
        # Default initialization creates None
        self.assertIsNone(s.scalar)
        
        s = Scalar(scalar=5.0)
        self.assertEqual(s.scalar.item(), 5.0)

    def test_scalar_tensor(self):
        """Test that scalar values are stored as tensors."""
        s = Scalar(scalar=5.0)
        self.assertTrue(torch.is_tensor(s.scalar))
        self.assertEqual(s.scalar.item(), 5.0)

    def test_scalar_set_get(self):
        """Test setting and getting scalar values."""
        s = Scalar()
        s.initialize(n_timesteps=5, batch_size=2)
        
        # Set a value
        s.set(3.5, step_index=0)
        self.assertEqual(s.get().item(), 3.5)
        
        # Set another value
        s.set(torch.tensor(7.2), step_index=0)
        self.assertAlmostEqual(s.get().item(), 7.2, places=5)

    def test_scalar_history(self):
        """Test history logging for scalars."""
        n_timesteps = 5
        s = Scalar()
        s.initialize(n_timesteps=5, batch_size=1)
        
        # Set multiple values to build history
        values = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=torch.float64)
        for i in range(n_timesteps):
            val = values[0,i]
            s.set(val, step_index=i)
        
        # Check history
        self.assertTrue(s._history_is_populated)
        history = s.history
        self.assertEqual(history.shape[1], 5)  # 5 timesteps

        torch.testing.assert_close(history, torch.tensor(values))

    def test_scalar_batch_dimensions(self):
        """Test scalar with batch dimensions."""
        s = Scalar()
        s.initialize(n_timesteps=10, batch_size=3)
        
        # Set batched values
        batched_value = torch.tensor([1.0, 2.0, 3.0])
        s.set(batched_value, step_index=0)
        
        result = s.get()
        self.assertEqual(result.shape[0], 3)  # batch_size
        torch.testing.assert_close(result, batched_value)

    def test_scalar_normalization(self):
        """Test scalar normalization."""
        s = Scalar()
        s.initialize(n_timesteps=5, batch_size=1)

        # Run 'simulation'
        for i in range(5):
            s.set(i, step_index=i)
        
        val = 3
        val_normalized = s.normalize(val)
        self.assertAlmostEqual(val_normalized.item(), (val - 0) / (4 - 0), places=5)

        val_denormalized = s.denormalize(val_normalized)
        self.assertAlmostEqual(val_denormalized.item(), val, places=5)
        


class TestVector(unittest.TestCase):
    def test_vector_initialization(self):
        """Test vector initialization."""
        v = Vector(size=2)
        self.assertEqual(v.size, 2)
        v.initialize(n_timesteps=10, batch_size=1)
        self.assertTrue(torch.is_tensor(v.get()))
        self.assertEqual(v.get().shape, (1, 2))  # Default batch_size=1

    def test_vector_set_get(self):
        """Test setting and getting vector values."""
        v = Vector(size=3)
        v.initialize(n_timesteps=5, batch_size=1)
        
        # Set a vector value
        test_vec = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
        v.set(test_vec, step_index=0)
        result = v.get()
        
        torch.testing.assert_close(result, test_vec)

    def test_vector_tensor_property(self):
        """Test vector tensor property and that size is overwritten by initialize."""
        v = Vector(size=4)
        v.initialize(n_timesteps=5, batch_size=1)
        v.set(torch.tensor([[1.0, 2.0, 3.0, 4.0]]), step_index=0)
        
        tensor = v.tensor
        self.assertTrue(torch.is_tensor(tensor))
        self.assertEqual(tensor.shape[1], 4)

    def test_vector_history(self):
        """Test history logging for vectors."""
        v = Vector(size=2)
        v.initialize(n_timesteps=5, batch_size=1)
        
        # Set multiple values to build history
        values = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]], dtype=torch.float64)
        for i in range(values.shape[1]):
            val = values[0,i,:]
            v.set(val, step_index=i)
        
        # Check history
        self.assertTrue(v._history_is_populated)
        history = v.history
        self.assertEqual(history.shape[1], 5)  # 5 timesteps
        self.assertEqual(history.shape[2], 2)  # size=2
        torch.testing.assert_close(history, torch.tensor(values, dtype=torch.float64))

    def test_vector_batch_dimensions(self):
        """Test vector with batch dimensions."""
        v = Vector(size=3)
        v.initialize(n_timesteps=10, batch_size=4)
        
        # Set batched values
        batched_value = torch.tensor([[1.0, 2.0, 3.0],
                                      [4.0, 5.0, 6.0],
                                      [7.0, 8.0, 9.0],
                                      [10.0, 11.0, 12.0]], dtype=torch.float64)
        v.set(batched_value, step_index=0)
        
        result = v.get()
        self.assertEqual(result.shape[0], 4)  # batch_size
        self.assertEqual(result.shape[1], 3)  # size
        torch.testing.assert_close(result, batched_value)


class TestParameter(unittest.TestCase):
    def test_parameter_initialization(self):
        """Test parameter initialization."""
        p = Parameter(torch.tensor(1.0))
        self.assertEqual(p.get().item(), 1.0)
        
    def test_parameter_bounds(self):
        """Test parameter bounds."""
        p = Parameter(torch.tensor(0.5), min_value=0.0, max_value=1.0)
        self.assertEqual(p.min_value.item(), 0.0)
        self.assertEqual(p.max_value.item(), 1.0)

    def test_parameter_set_get(self):
        """Test setting and getting parameter values."""
        p = Parameter(torch.tensor(1.0), min_value=0.0, max_value=10.0)
        
        # Set a new value
        p.set(5.0, normalized=False)
        self.assertAlmostEqual(p.get().item(), 5.0, places=5)
        
        # Set normalized value (0.5 should map to 5.0)
        p.set(0.5, normalized=True)
        self.assertAlmostEqual(p.get().item(), 5.0, places=5)

    def test_parameter_tensor_property(self):
        """Test parameter tensor property."""
        p = Parameter(torch.tensor(2.5), min_value=0.0, max_value=2.5)
        self.assertTrue(torch.is_tensor(p))
        self.assertEqual(p.get(), 2.5)
        self.assertEqual(p.item(), 1.0)

    def test_parameter_gradient_computation(self):
        """Test that parameters support gradient computation."""
        p = Parameter(torch.tensor(1.0), requires_grad=True)
        
        # Check that gradients can be computed
        self.assertTrue(p.requires_grad)
        
        # Simple computation
        y = p ** 2
        y.backward()
        
        # Gradient of x^2 at x=1 is 2x = 2
        self.assertIsNotNone(p.grad)
        self.assertAlmostEqual(p.grad.item(), 2.0, places=5)

    def test_parameter_batch_dimensions(self):
        """Test parameter with batch dimensions."""
        p = Parameter(torch.tensor([1.0, 2.0, 3.0]))
        
        result = p.get()
        self.assertEqual(result.shape[0], 3)
        torch.testing.assert_close(result, torch.tensor([1.0, 2.0, 3.0]))


if __name__ == '__main__':
    unittest.main()

    v = TestVector()
    v.test_vector_history()

