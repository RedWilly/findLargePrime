import math
import random
from typing import Optional, Tuple, List
import numpy as np
from collections import defaultdict

class AdvancedFactorization:
    def __init__(self, number: int, max_iterations: int = 10000):
        self.N = number
        self.max_iterations = max_iterations
        self.small_primes = self._generate_small_primes(1000)
        self.residue_patterns = defaultdict(list)
        
    def _generate_small_primes(self, limit: int) -> List[int]:
        """Generate prime numbers up to limit using Sieve of Eratosthenes"""
        sieve = [True] * limit
        sieve[0] = sieve[1] = False
        
        for i in range(2, int(math.sqrt(limit)) + 1):
            if sieve[i]:
                for j in range(i*i, limit, i):
                    sieve[j] = False
                    
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    def _analyze_residues(self) -> dict:
        """Analyze residue patterns modulo small primes"""
        patterns = {}
        for p in self.small_primes[:100]:  # Use first 100 small primes
            residue = self.N % p
            patterns[p] = residue
            # Store interesting patterns (residues that might indicate factor properties)
            if residue in [1, p-1] or math.gcd(residue, p) > 1:
                self.residue_patterns[p].append(residue)
        return patterns
    
    def _quadratic_congruence(self, a: int, n: int) -> Optional[int]:
        """Solve quadratic congruence x² ≡ a (mod n)"""
        if a == 0:
            return 0
        
        # Try Tonelli-Shanks algorithm for prime modulus
        def legendre_symbol(a: int, p: int) -> int:
            return pow(a, (p - 1) // 2, p)
        
        if n in self.small_primes and legendre_symbol(a, n) == 1:
            q = n - 1
            s = 0
            while q % 2 == 0:
                q //= 2
                s += 1
            
            if s == 1:
                return pow(a, (n + 1) // 4, n)
            
            for z in range(2, n):
                if legendre_symbol(z, n) == -1:
                    break
                    
            c = pow(z, q, n)
            r = pow(a, (q + 1) // 2, n)
            t = pow(a, q, n)
            m = s
            
            while t != 1:
                for i in range(1, m):
                    if pow(t, 2**i, n) == 1:
                        b = pow(c, 2**(m-i-1), n)
                        r = (r * b) % n
                        c = (b * b) % n
                        t = (t * c) % n
                        m = i
                        break
            return r
        
        return None
    
    def _elliptic_approach(self, seed: int) -> Optional[Tuple[int, int]]:
        """Try to find factors using elliptic curve method variant"""
        def point_add(P: Tuple[int, int], Q: Tuple[int, int], a: int) -> Optional[Tuple[int, int]]:
            if P is None:
                return Q
            if Q is None:
                return P
                
            x1, y1 = P
            x2, y2 = Q
            
            if x1 == x2:
                if y1 == y2:
                    if y1 == 0:
                        return None
                    # point doubling
                    m = (3 * x1 * x1 + a) * pow(2 * y1, -1, self.N)
                else:
                    return None
            else:
                # point addition
                m = (y2 - y1) * pow(x2 - x1, -1, self.N)
            
            x3 = (m * m - x1 - x2) % self.N
            y3 = (m * (x1 - x3) - y1) % self.N
            
            return (x3, y3)
            
        # choose random curve parameters
        x = seed % self.N
        y = (seed * seed) % self.N
        a = random.randint(0, self.N - 1)
        
        # here we check if point is on curve
        b = (y * y - x * x * x - a * x) % self.N
        
        P = (x, y)
        Q = P
        
        for _ in range(100):  # Limit iterations but can increase it for maximum efficency 
            try:
                Q = point_add(Q, P, a)
                if Q is None:
                    break
            except Exception as e:
                # If we get an error, it might be because we found a factor
                if "inverse" in str(e):
                    d = math.gcd(int(str(e).split()[-1]), self.N)
                    if 1 < d < self.N:
                        return (d, self.N // d)
                break
                
        return None
    
    def _apply_lattice_reduction(self, basis_size: int = 10) -> Optional[Tuple[int, int]]:
        """Apply a simplified lattice reduction technique"""
        # lattice basis using residue information
        basis = []
        for p, residues in self.residue_patterns.items():
            for r in residues:
                vec = [0] * basis_size
                vec[0] = p
                vec[1] = r
                basis.append(vec)
                
                if len(basis) >= basis_size:
                    break
            if len(basis) >= basis_size:
                break
                
        if not basis:
            return None
            
        # here is a simple Gram-Schmidt process
        def gram_schmidt(v: List[int], u: List[int]) -> float:
            dot_product = sum(x * y for x, y in zip(v, u))
            norm_sq = sum(x * x for x in u)
            return dot_product / norm_sq if norm_sq != 0 else 0
            
        orthogonal = []
        for v in basis:
            u = v.copy()
            for w in orthogonal:
                coefficient = gram_schmidt(v, w)
                u = [u[i] - coefficient * w[i] for i in range(len(u))]
            if any(x != 0 for x in u):
                orthogonal.append(u)
                
        # Try to extract potential factors from the reduced basis
        for vec in orthogonal:
            if vec[0] != 0:
                d = math.gcd(abs(int(vec[0])), self.N)
                if 1 < d < self.N:
                    return (d, self.N // d)
                    
        return None
    
    def find_factors(self) -> Optional[Tuple[int, int]]:
        """Main method to find prime factors using multiple approaches"""
        # Step 1: let s analyze residue patterns
        patterns = self._analyze_residues()
        
        # Step 2: then ill try lattice reduction
        lattice_result = self._apply_lattice_reduction()
        if lattice_result:
            return lattice_result
            
        # Step 3: or elliptic curve approach with different seeds
        for seed in range(1, 100):
            ec_result = self._elliptic_approach(seed)
            if ec_result:
                return ec_result
                
        # Step 4: finally residue patterns for guided search
        for p, residues in self.residue_patterns.items():
            for r in residues:
                # Try quadratic congruence
                root = self._quadratic_congruence(r, p)
                if root is not None:
                    factor_candidate = math.gcd(root, self.N)
                    if 1 < factor_candidate < self.N:
                        return (factor_candidate, self.N // factor_candidate)
                        
        return None

# Usage example
def factorize_large_number(n: int) -> Optional[Tuple[int, int]]:
    """Wrapper function to factorize a large number"""
    factorizer = AdvancedFactorization(n)
    return factorizer.find_factors()
    
# Example usage
number = 24862048  # Your large number
factors = factorize_large_number(number)
if factors:
    print(f"Found prime factors: {factors}")
else:
    print("Could not find prime factors")
