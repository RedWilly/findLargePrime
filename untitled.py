import math
import numpy as np
from typing import List, Tuple, Optional, Dict
from decimal import Decimal, getcontext

# Set precision for decimal calculations
getcontext().prec = 100

class NumberTheoreticFactorization:
    def __init__(self, n: int):
        self.N = n
        self.small_primes = self._sieve_of_eratosthenes(1000)
        self.residue_data: Dict[int, int] = {}
        
    def _sieve_of_eratosthenes(self, limit: int) -> List[int]:
        """Generate primes up to limit using Sieve of Eratosthenes"""
        sieve = [True] * limit
        sieve[0] = sieve[1] = False
        for i in range(2, int(math.sqrt(limit)) + 1):
            if sieve[i]:
                for j in range(i * i, limit, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    def _compute_residues(self) -> Dict[int, Dict[str, any]]:
        """
        Compute and analyze residues modulo small primes
        Returns detailed analysis including quadratic residue information
        """
        residue_analysis = {}
        for p in self.small_primes:
            r = self.N % p
            self.residue_data[p] = r
            
            # Compute Legendre symbol for quadratic residue testing
            legendre = self._legendre_symbol(self.N, p) if p > 2 else 0
            
            residue_analysis[p] = {
                'residue': r,
                'is_quadratic_residue': legendre == 1,
                'legendre_symbol': legendre,
                'relative_size': r / p
            }
        return residue_analysis
    
    def _legendre_symbol(self, a: int, p: int) -> int:
        """Compute the Legendre symbol (a/p)"""
        if p < 2:
            raise ValueError("p must be prime")
        if a == 0:
            return 0
        if a == 1:
            return 1
            
        if a % 2 == 0:
            return self._legendre_symbol(a // 2, p) * (-1)**((p**2 - 1) // 8)
            
        return self._legendre_symbol(p % a, a) * (-1)**((a - 1) * (p - 1) // 4)
    
    def _continued_fraction_sqrt(self, max_iterations: int = 100) -> List[Tuple[int, List[int]]]:
        """
        Compute continued fraction expansion of sqrt(N)
        Returns list of convergents and the expansion terms
        """
        def floor_sqrt(n: int) -> int:
            x = n
            y = (x + 1) // 2
            while y < x:
                x = y
                y = (x + n // x) // 2
            return x
            
        a0 = floor_sqrt(self.N)
        if a0 * a0 == self.N:
            return [(a0, [a0])]
            
        convergents = []
        m = 0
        d = 1
        a = a0
        expansion = [a0]
        
        # Initialize convergent calculations
        h_minus_2, h_minus_1 = 0, 1
        k_minus_2, k_minus_1 = 1, 0
        
        for _ in range(max_iterations):
            m = d * a - m
            d = (self.N - m * m) // d
            if d == 0:
                break
            a = (a0 + m) // d
            expansion.append(a)
            
            # Calculate convergent
            h = a * h_minus_1 + h_minus_2
            k = a * k_minus_1 + k_minus_2
            
            convergents.append((h, expansion.copy()))
            
            # Update values for next iteration
            h_minus_2, h_minus_1 = h_minus_1, h
            k_minus_2, k_minus_1 = k_minus_1, k
            
            # Check for periodicity
            if d == 1 and a == 2 * a0:
                break
                
        return convergents
    
    def _construct_lattice(self, dimension: int = 4) -> np.ndarray:
        """
        Construct a lattice based on residue information and N
        Returns the lattice basis matrix
        """
        basis = np.zeros((dimension, dimension), dtype=np.int64)
        
        # First row represents the relation x + y = N
        basis[0][0] = 1
        basis[0][1] = 1
        basis[0][2] = -self.N
        
        # Use residue information for additional constraints
        sorted_residues = sorted(
            self.residue_data.items(),
            key=lambda x: abs(x[1] - self.N // x[0]) if x[0] != 0 else float('inf')
        )
        
        for i in range(1, dimension):
            if i - 1 < len(sorted_residues):
                p, r = sorted_residues[i - 1]
                basis[i][0] = p
                basis[i][1] = r
                basis[i][i] = 1
                
        return basis
    
    def _gram_schmidt(self, basis: np.ndarray) -> np.ndarray:
        """
        Perform Gram-Schmidt orthogonalization
        Returns orthogonalized basis
        """
        n = basis.shape[0]
        orthogonal = np.zeros_like(basis, dtype=np.float64)
        orthogonal[0] = basis[0]
        
        for i in range(1, n):
            orthogonal[i] = basis[i]
            for j in range(i):
                if np.any(orthogonal[j]):
                    projection = np.dot(basis[i], orthogonal[j]) / np.dot(orthogonal[j], orthogonal[j])
                    orthogonal[i] = orthogonal[i] - projection * orthogonal[j]
                    
        return orthogonal
    
    def _analyze_convergent(self, h: int, k: int) -> Optional[Tuple[int, int]]:
        """Analyze a convergent for potential factors"""
        if k == 0:
            return None
            
        # Check if h²/k² is close to N
        approx = h * h // (k * k)
        if abs(approx - self.N) < math.sqrt(self.N):
            # Try to factor using this approximation
            potential_factor = math.gcd(h + k, self.N)
            if 1 < potential_factor < self.N:
                return (potential_factor, self.N // potential_factor)
        return None

  #3 step
    def find_factors(self) -> Optional[Tuple[int, int]]:
        """
        Main method to find factors using multiple approaches
        Returns tuple of factors if found, None otherwise
        """
        # Step 1: Analyze residues
        residue_analysis = self._compute_residues()
        
        # Check for immediate factors from residues
        for p, analysis in residue_analysis.items():
            if analysis['residue'] == 0:
                if self.N % p == 0:
                    return (p, self.N // p)
        
        # Step 2: Try continued fraction approach
        convergents = self._continued_fraction_sqrt()
        for h, _ in convergents:
            k = int(math.sqrt(h * h // self.N))
            result = self._analyze_convergent(h, k)
            if result:
                return result
        
        # Step 3: Try lattice-based approach
        basis = self._construct_lattice()
        orthogonal = self._gram_schmidt(basis)
        
        # Look for short vectors that might indicate factors
        for row in orthogonal:
            if np.any(row):
                norm = np.linalg.norm(row)
                if norm < math.sqrt(self.N):
                    # Try to extract factors from the short vector
                    potential_factor = math.gcd(int(abs(row[0])), self.N)
                    if 1 < potential_factor < self.N:
                        return (potential_factor, self.N // potential_factor)
        
        return None

def factorize_number(n: int) -> Optional[Tuple[int, int]]:
    """Wrapper function to factorize a number"""
    factorizer = NumberTheoreticFactorization(n)
    return factorizer.find_factors()

# Example usage and testing
def test_algorithm():
    """Test the algorithm with various numbers"""
    test_cases = [
        299,    # 13 × 23
        10403,  # 101 × 103
        8051,   # 83 × 97
    ]
    
    for n in test_cases:
        print(f"\nTesting number: {n}")
        factors = factorize_number(n)
        if factors:
            print(f"Found factors: {factors}")
            print(f"Verification: {factors[0]} × {factors[1]} = {factors[0] * factors[1]}")
            assert factors[0] * factors[1] == n
        else:
            print("No factors found")
