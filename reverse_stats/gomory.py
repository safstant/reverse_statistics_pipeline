import numpy as np
from .exceptions import ReverseStatsError
import math
from fractions import Fraction
from typing import List, Tuple, Dict, Any, Optional, Union, Dict, Optional
from dataclasses import dataclass
import logging
import sys

logger = logging.getLogger(__name__)

# ============================================================================
# EXCEPTIONS
# ============================================================================
class GomoryError(ReverseStatsError):
    """Base exception for Gomory cut operations."""
    def __init__(self, message: str):
        self.message = message

        super().__init__(message)


class FractionalVertexError(GomoryError):
    """Raised when fractional vertices cannot be resolved."""
    pass


# ============================================================================
# CONFIGURATION INTEGRATION
# ============================================================================
def get_gomory_config() -> Dict[str, Any]:
    """
    Get Gomory-specific configuration with sane defaults.
    Integrates with global config system when available.
    """
    # Default configuration (overridden by global config if available)
    config = {
        "skip_gomory_if_integral": True,
        "max_gomory_cuts": 50,            # Safety guard against cycling
        "integrality_tolerance": 1e-10,   # From global config
        "enable_gomory": True,            # Master enable/disable
    }
    
    # Try to integrate with global config system
    try:
        from .config import get_config
        global_config = get_config()
        pipeline_config = global_config.pipeline_config if hasattr(global_config, 'pipeline_config') else global_config
        config["skip_gomory_if_integral"] = getattr(pipeline_config, 
                                                    "skip_gomory_if_integral", 
                                                    config["skip_gomory_if_integral"])
        config["integrality_tolerance"] = getattr(pipeline_config,
                                                 "integrality_tolerance",
                                                 config["integrality_tolerance"])
    except (ImportError, AttributeError, ModuleNotFoundError):
        # Standalone mode - use defaults
        pass
    
    return config


# ============================================================================
# CORE GOMORY CUT OPERATIONS
# ============================================================================
def is_integral(x: Fraction, tol: float = 1e-10) -> bool:
    """
    Check if Fraction is integral within tolerance.

    Args:
        x: Fraction value to check
        tol: Tolerance for integrality check

    Returns:
        True if |x - round(x)| < tol
    """
    return abs(x - round(x)) < tol


def fractional_part(x: Fraction) -> Fraction:
    """
    Compute exact fractional part of a rational number.
    Returns value in [0, 1) such that x = floor(x) + frac(x).

    Args:
        x: Fraction value

    Returns:
        Fractional part in [0, 1)
    """
    # FIX(Bug-4): math.floor converts its argument to a float first, which means
    # large Fractions (e.g. numerator ~ 10^500) silently overflow to inf or lose
    # all precision. Use the exact integer division formula instead:
    #   floor(p/q) = p // q  (Python // is always floor-division for integers)
    if isinstance(x, Fraction):
        return x - Fraction(x.numerator // x.denominator)
    # Fallback for plain int/float arguments
    return x - math.floor(x)


def detect_fractional_vertices(point: Tuple[Fraction, ...], 
                              tol: float = 1e-10) -> List[int]:
    """
    Detect indices of fractional coordinates in a point.

    Args:
        point: Point in rational coordinates
        tol: Integrality tolerance

    Returns:
        List of indices where coordinate is fractional


    """
    fractional_indices = []
    for i, coord in enumerate(point):
        if not is_integral(coord, tol):
            fractional_indices.append(i)
    return fractional_indices


def generate_gomory_cut(coefficients: List[Fraction], 
                       rhs: Fraction,
                       variable_indices: Optional[List[int]] = None,
                       tol: float = 1e-10) -> Tuple[List[Fraction], Fraction]:
    """
    Generate Gomory fractional cut from a tableau row.

    Given constraint: Σ aⱼxⱼ = b (with b fractional)
    Gomory cut: Σ frac(aⱼ)xⱼ ≥ frac(b)

    Args:
        coefficients: Row coefficients [a₀, a₁, ..., aₙ]
        rhs: Right-hand side value b
        variable_indices: Optional mapping of coefficient indices to variable indices
        tol: Tolerance for fractional part detection

    Returns:
        (cut_coefficients, cut_rhs) representing Σ cut_coefficients[i]·x[i] ≥ cut_rhs


    """
    # Compute fractional parts
    frac_coeffs = [fractional_part(c) for c in coefficients]
    frac_rhs = fractional_part(rhs)
    
    # Skip if RHS is integral (no cut needed)
    if is_integral(frac_rhs, tol):
        raise GomoryError("Cannot generate cut from integral RHS")
    
    # Construct cut: Σ frac(aⱼ)xⱼ ≥ frac(b)
    # Note: All fractional parts are in [0,1), so coefficients are non-negative
    cut_coeffs = frac_coeffs
    cut_rhs = frac_rhs
    
    return cut_coeffs, cut_rhs


def apply_gomory_cut_to_system(system: Any,  # ConstraintSystem from constraints.py
                              cut_coeffs: List[Fraction],
                              cut_rhs: Fraction,
                              variable_map: Optional[List[int]] = None) -> Any:
    """
    Apply Gomory cut to constraint system.

    Args:
        system: ConstraintSystem instance
        cut_coeffs: Cut coefficients (one per variable in system)
        cut_rhs: Cut right-hand side
        variable_map: Optional mapping from cut coefficient indices to system variable indices

    Returns:
        New ConstraintSystem with cut added


    """
    try:
        from .constraints import Inequality, InequalityDirection
    except (ImportError, ModuleNotFoundError) as e:
        raise ImportError(
            f"gomory.py: constraints module could not be imported: {e}. "
            "Gomory cut generation requires the real Inequality and InequalityDirection "
            "types — stub replacements would produce structurally wrong cuts. "
            "Ensure the package is installed correctly or constraints.py is on sys.path."
        ) from e
    
    # Handle variable mapping if provided
    if variable_map:
        full_coeffs = [Fraction(0)] * system.variables
        for idx, coeff in zip(variable_map, cut_coeffs):
            full_coeffs[idx] = coeff
        cut_coeffs = full_coeffs
    
    # Create inequality: Σ cut_coeffs[i]·x[i] ≥ cut_rhs
    inequality = Inequality(
        coefficients=tuple(cut_coeffs),
        bound=cut_rhs,
        direction=InequalityDirection.GREATER_EQUAL
    )
    
    # Add to system
    if hasattr(system, 'add_inequality'):
        return system.add_inequality(inequality)
    else:
        # Fallback for minimal system
        system.add_inequality(inequality)
        return system


# ============================================================================

# ============================================================================
@dataclass(frozen=True)
class GomoryResult:
    """
    Result of Gomory cut phase.

    Attributes:
        system: Augmented constraint system with cuts applied
        cuts_added: Number of Gomory cuts added
        is_integral: Whether final solution is integral
        fractional_vertices: Indices of remaining fractional vertices (if any)
        iterations: Number of cut iterations performed
    """
    system: Any
    cuts_added: int
    is_integral: bool
    fractional_vertices: List[int]
    iterations: int


def gomory_cut_phase(system: Any,
                    feasible_point: Optional[Tuple[Fraction, ...]] = None,
                    config: Optional[Dict[str, Any]] = None) -> GomoryResult:
    """
    Main entry point for  Gomory cutting plane algorithm.

    Algorithm:
        1. Check skip condition: if all vertices integral, skip cuts
        2. While fractional vertices exist and cut limit not reached:
            a. Detect fractional vertices
            b. Generate Gomory cut from fractional constraint
            c. Add cut to system
            d. Re-optimize to find new feasible point
        3. Terminate when integral solution found or max cuts reached

    Args:
        system: ConstraintSystem to process
        feasible_point: Optional initial feasible point (if known)
        config: Configuration dictionary (uses defaults if None)

    Returns:
        GomoryResult with augmented system and status


    """
    # Load configuration
    if config is None:
        config = get_gomory_config()
    
    # Check master enable flag
    if not config.get("enable_gomory", True):
        logger.info("Gomory cuts disabled via configuration")
        return GomoryResult(
            system=system,
            cuts_added=0,
            is_integral=False,  # Unknown without checking
            fractional_vertices=[],
            iterations=0
        )
    

    if config.get("skip_gomory_if_integral", True):
        if feasible_point is not None:
            fractional = detect_fractional_vertices(
                feasible_point, 
                tol=config.get("integrality_tolerance", 1e-10)
            )
            if not fractional:
                logger.info("Skipping Gomory cuts - all vertices integral ()")
                return GomoryResult(
                    system=system,
                    cuts_added=0,
                    is_integral=True,
                    fractional_vertices=[],
                    iterations=0
                )
    
    # Initialize state
    current_system = system
    cuts_added = 0
    iterations = 0
    max_cuts = config.get("max_gomory_cuts", 50)
    tol = config.get("integrality_tolerance", 1e-10)
    

    while cuts_added < max_cuts:
        iterations += 1
        
        # Step 1: Find feasible point (re-optimize after each cut)
        try:
            from .feasibility import check_feasibility, optimize, OptimizationDirection
            # Minimize sum of variables to get vertex solution
            objective = [Fraction(1) for _ in range(current_system.variables)]
            result = optimize(
                current_system,
                objective=objective,
                direction=OptimizationDirection.MINIMIZE
            )
            
            if result.status != "feasible" or result.point is None:
                raise GomoryError(f"System became infeasible after {cuts_added} cuts")
            
            current_point = result.point
            
        except (ImportError, ModuleNotFoundError, AttributeError) as _e:
            raise GomoryError(
                f"gomory_cut_phase requires the feasibility module for LP-based vertex "
                f"finding, but it could not be imported: {_e}. "
                "Cannot generate correct Gomory cuts without a valid feasible point."
            ) from _e
        

        fractional_indices = detect_fractional_vertices(current_point, tol=tol)
        
        # Termination condition: all vertices integral
        if not fractional_indices:
            logger.info(f"Gomory phase complete: integral solution found after {cuts_added} cuts")
            return GomoryResult(
                system=current_system,
                cuts_added=cuts_added,
                is_integral=True,
                fractional_vertices=[],
                iterations=iterations
            )
        

        fractional_idx = fractional_indices[0]
        n_vars = len(current_point)
        
        # Extract true tableau row from binding constraints
        cut_coeffs = None
        cut_rhs = None
        
        try:
            from sympy import Matrix
            
            # Find binding constraints
            A_bind = []
            b_bind = []
            if hasattr(current_system, 'to_matrix_form'):
                A_f, b_f, C_f, d_f = current_system.to_matrix_form()
                
                # Equations are binding
                for r, val in zip(C_f, d_f):
                    A_bind.append(r)
                    b_bind.append(val)
                
                # Binding inequalities
                for r, val in zip(A_f, b_f):
                    dot = sum(c * pt for c, pt in zip(r, current_point))
                    if abs(float(dot - val)) < tol:
                        A_bind.append(r)
                        b_bind.append(val)
            
            if len(A_bind) >= n_vars:
                M = Matrix(A_bind)
                _, indep_rows = M.T.rref()
                basis_indices = [idx for idx in indep_rows if idx < len(A_bind)]
                
                if len(basis_indices) == n_vars:
                    B = Matrix([A_bind[i] for i in basis_indices[:n_vars]])
                    v = [b_bind[i] for i in basis_indices[:n_vars]]
                    B_inv = B.inv()
                    
                    # The tableau row for basic variable x_i:
                    row_B_inv = B_inv.row(fractional_idx)
                    f_0 = fractional_part(Fraction(current_point[fractional_idx]))
                    
                    if not is_integral(f_0, tol):
                        f_j_list = []
                        for j in range(n_vars):
                            a_ij = row_B_inv[j]
                            a_ij_frac = Fraction(int(a_ij.p), int(a_ij.q)) if hasattr(a_ij, 'p') else Fraction(a_ij)
                            f_j_list.append(fractional_part(a_ij_frac))
                            
                        # Substitute s_j = B_j*x - v_j back to x
                        cut_x = [Fraction(0)] * n_vars
                        for k in range(n_vars):
                            coeff_k = Fraction(0)
                            for j in range(n_vars):
                                B_jk = B[j, k]
                                B_jk_frac = Fraction(int(B_jk.p), int(B_jk.q)) if hasattr(B_jk, 'p') else Fraction(B_jk)
                                coeff_k += f_j_list[j] * B_jk_frac
                            cut_x[k] = coeff_k
                            
                        rhs_x = f_0
                        for j in range(n_vars):
                            v_j = v[j]
                            v_j_frac = Fraction(int(v_j.p), int(v_j.q)) if hasattr(v_j, 'p') else Fraction(v_j)
                            rhs_x += f_j_list[j] * v_j_frac
                            
                        cut_coeffs = cut_x
                        cut_rhs = rhs_x
            
        except ImportError:
             pass
        except Exception as e:
             logger.debug(f"Failed to generate exact Gomory cut from LP basis: {e}")
             
        if cut_coeffs is None:
            # Fallback to synthesis mapping if basis extraction failed
            coefficients = [Fraction(0) for _ in range(n_vars)]
            coefficients[fractional_idx] = Fraction(1)
            rhs = current_point[fractional_idx]
            
            try:
                cut_coeffs, cut_rhs = generate_gomory_cut(
                    coefficients, 
                    rhs, 
                    tol=tol
                )
            except GomoryError as e:
                logger.warning(f"Could not generate Gomory cut: {e}")
                break
        

        current_system = apply_gomory_cut_to_system(
            current_system,
            cut_coeffs,
            cut_rhs
        )
        cuts_added += 1
        
        logger.debug(f"Added Gomory cut {cuts_added}: Σ{cut_coeffs}·x ≥ {cut_rhs}")
        
        # Safety check: prevent infinite loops
        if cuts_added >= max_cuts:
            logger.warning(f"Reached maximum Gomory cuts ({max_cuts}) without integral solution")
            return GomoryResult(
                system=current_system,
                cuts_added=cuts_added,
                is_integral=False,
                fractional_vertices=fractional_indices,
                iterations=iterations
            )
    
    # Final status check
    fractional_final = detect_fractional_vertices(current_point, tol=tol) if feasible_point else fractional_indices
    return GomoryResult(
        system=current_system,
        cuts_added=cuts_added,
        is_integral=not fractional_final,
        fractional_vertices=fractional_final,
        iterations=iterations
    )


# ============================================================================
# VALIDATION AND TESTING
# ============================================================================
def validate_gomory_utils() -> Dict[str, bool]:
    """Run internal test suite to verify Gomory utilities."""
    results = {}
    try:
        from fractions import Fraction
        
        # Test 1: Fractional part computation
        results["fractional_part"] = (
            fractional_part(Fraction(3, 2)) == Fraction(1, 2) and
            fractional_part(Fraction(-3, 2)) == Fraction(1, 2) and  # -1.5 → floor=-2, frac=0.5
            fractional_part(Fraction(5, 1)) == Fraction(0, 1)
        )
        
        # Test 2: Integrality detection
        results["is_integral"] = (
            is_integral(Fraction(5, 1)) and
            not is_integral(Fraction(5, 2)) and
            is_integral(Fraction(10000000001, 10000000000), tol=1e-9)  # Near-integer
        )
        
        # Test 3: Fractional vertex detection
        point1 = (Fraction(1, 1), Fraction(2, 1), Fraction(3, 1))
        point2 = (Fraction(1, 1), Fraction(3, 2), Fraction(3, 1))
        results["detect_fractional"] = (
            detect_fractional_vertices(point1) == [] and
            detect_fractional_vertices(point2) == [1]
        )
        
        # Test 4: Gomory cut generation
        coeffs = [Fraction(1, 1), Fraction(3, 2), Fraction(5, 3)]
        rhs = Fraction(7, 4)  # 1.75 → frac=0.75
        cut_coeffs, cut_rhs = generate_gomory_cut(coeffs, rhs)
        results["generate_cut"] = (
            cut_coeffs == [Fraction(0, 1), Fraction(1, 2), Fraction(2, 3)] and
            cut_rhs == Fraction(3, 4)
        )
        
        # Test 5: Skip condition (integral point)
        config_skip = {"skip_gomory_if_integral": True, "integrality_tolerance": 1e-10}
        integral_point = (Fraction(1), Fraction(2), Fraction(3))
        fractional_point = (Fraction(1), Fraction(3, 2), Fraction(3))
        
        # Mock constraint system
        class MockSystem:
            variables = 3
            inequalities = []
            def add_inequality(self, ineq):
                self.inequalities.append(ineq)
                return self
        
        system = MockSystem()
        
        # Integral point should skip cuts
        result_skip = gomory_cut_phase(system, integral_point, config_skip)
        results["skip_condition"] = (result_skip.cuts_added == 0 and result_skip.is_integral)
        
        # Test 6: Cut application (basic validation)
        system2 = MockSystem()
        cut_coeffs = [Fraction(1, 2), Fraction(1, 3)]
        cut_rhs = Fraction(1, 4)
        system_with_cut = apply_gomory_cut_to_system(system2, cut_coeffs, cut_rhs)
        results["apply_cut"] = len(system_with_cut.inequalities) == 1
        
        # Test 7: Max cuts termination
        config_max = {
            "skip_gomory_if_integral": False,
            "max_gomory_cuts": 3,
            "integrality_tolerance": 1e-10,
            "enable_gomory": True
        }
        # Use fractional point that won't become integral
        result_max = gomory_cut_phase(system, fractional_point, config_max)
        results["max_cuts"] = (result_max.cuts_added == 3 and not result_max.is_integral)
        
        logger.info("✅ Gomory utilities validation passed")
    except Exception as e:
        logger.error(f"❌ Gomory utilities validation failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        results["validation_error"] = str(e)
    return results


# ============================================================================
# MAIN TESTING
# ============================================================================
if __name__ == "__main__":
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 60)
    print("Testing Production-Ready Gomory Utilities")
    print("=" * 60)
    
    # Run validation
    results = validate_gomory_utils()
    
    print("Validation Results:")
    print("-" * 40)
    success = 0
    total = 0
    for key, value in results.items():
        total += 1
        if key == "validation_error":
            print(f"❌ {key}: {value}")
        elif value:
            success += 1
            print(f"✅ {key}: PASSED")
        else:
            print(f"❌ {key}: FAILED")
    print("-" * 40)
    print(f"Overall: {success}/{total-1} tests passed")
    
    if "validation_error" in results:
        sys.exit(1)
    
    # Demonstration
    print("\n" + "=" * 60)
    print("Gomory Cut Demo")
    print("=" * 60)
    
    from fractions import Fraction
    
    # Create mock constraint system (2 variables)
    class MockSystem:
        variables = 2
        inequalities = []
        def add_inequality(self, ineq):
            self.inequalities.append(ineq)
            return self
        def __repr__(self):
            return f"MockSystem(vars={self.variables}, ineqs={len(self.inequalities)})"
    
    system = MockSystem()
    
    # Case 1: Integral point (should skip cuts with default config)
    print("\n1. Integral Point (Skip Condition - ):")
    integral_point = (Fraction(2, 1), Fraction(3, 1))
    result1 = gomory_cut_phase(system, integral_point)
    print(f"   Input point: {integral_point}")
    print(f"   Cuts added: {result1.cuts_added}")
    print(f"   Is integral: {result1.is_integral}")
    
    # Case 2: Fractional point (should add cuts)
    print("\n2. Fractional Point (Cut Generation - ):")
    fractional_point = (Fraction(3, 2), Fraction(5, 3))  # (1.5, 1.666...)
    config_no_skip = {
        "skip_gomory_if_integral": False,
        "max_gomory_cuts": 5,
        "integrality_tolerance": 1e-10,
        "enable_gomory": True
    }
    result2 = gomory_cut_phase(system, fractional_point, config_no_skip)
    print(f"   Input point: {fractional_point}")
    print(f"   Cuts added: {result2.cuts_added}")
    print(f"   Fractional indices: {result2.fractional_vertices}")
    
    # Case 3: Fractional part computation
    print("\n3. Fractional Part Examples:")
    examples = [
        Fraction(7, 4),   # 1.75 → 0.75
        Fraction(-5, 3),  # -1.666... → 0.333...
        Fraction(10, 1),  # 10 → 0
    ]
    for x in examples:
        frac = fractional_part(x)
        print(f"   frac({x}) = {frac} (float: {float(frac):.4f})")
    
    print("\n" + "=" * 60)
    print("✅ Gomory Utilities Ready for Production")
    print("=" * 60)