# Math Numerical Methods

This repository, specifically the `Random Problems/Math` directory, contains Jupyter notebooks exploring fundamental concepts and applications of numerical methods in mathematics. The primary focus is on demonstrating different numerical techniques for solving common problems, including interpolation, numerical differentiation, and solving partial differential equations.

## Project Structure

This project is organized within the `Random Problems/Math` directory. The main component is the `Numerical.ipynb` Jupyter notebook.

## `Numerical.ipynb` Overview

The `Numerical.ipynb` notebook provides a comprehensive exploration of several core numerical analysis topics. It is structured into distinct parts, each addressing a specific problem or concept.

### 1. Polynomial Interpolation

This section demonstrates two methods for finding an interpolating polynomial for a given set of data points:
*   **Method 1: Linear System of Equations (Vandermonde Matrix)**
    *   **Purpose**: To determine the coefficients of an interpolating polynomial by setting up and solving a linear system of equations.
    *   **Logic**: The polynomial `P(x) = a_0 + a_1 x + ... + a_n x^n` is fitted to `n+1` data points `(x_i, y_i)`. This forms a system `V * a = y`, where `V` is the Vandermonde matrix. `NumPy`'s `vander` and `linalg.solve` are used to compute the coefficients.
*   **Method 2: Lagrange Interpolation Formula**
    *   **Purpose**: To construct the interpolating polynomial directly using Lagrange basis polynomials.
    *   **Logic**: The polynomial is expressed as a sum `P(x) = Σ y_i * l_i(x)`, where `l_i(x)` are the Lagrange basis polynomials. Custom Python functions are implemented to build these basis functions and the final polynomial.
*   **Comparison (Symbolic and Graphical)**
    *   **Purpose**: To verify that both methods yield the same interpolating polynomial and to visualize their behavior.
    *   **Logic**: `SymPy` is used to derive symbolic expressions for the polynomials from both methods. These symbolic forms are then compared, and their numerical evaluations are plotted on the same graphs across various ranges (interpolation and extrapolation) to show their agreement.

### 2. Numerical Differencing Schemes

This part focuses on deriving and presenting formulas for approximating derivatives using finite differences.
*   **Part 1: 8th-Order Centered Finite Difference for First Derivative**
    *   **Purpose**: To develop a highly accurate numerical scheme for the first derivative `df(x)/dx`.
    *   **Logic**: A 9-point centered stencil (from `x-4h` to `x+4h`) is used. The formula is derived by applying Taylor series expansions and solving for coefficients to achieve an `O(h^8)` order of accuracy.
*   **Part 2: Mixed Derivative `∂³f / (∂x²∂y)` Approximation**
    *   **Purpose**: To construct a consistent finite difference approximation for a mixed third-order partial derivative of a function `f(x,y)`.
    *   **Logic**: The formula is derived by applying a second-order central difference approximation for `∂²/∂x²` to a first-order central difference approximation for `∂/∂y`. The resulting scheme is accurate to `O(Δx², Δy²)`.

### 3. Solving 2D Laplace Equation with Jacobi Iteration

This section demonstrates the numerical solution of a 2D Laplace equation with given boundary conditions using an iterative method.
*   **Problem**: Solving `∂²u/∂x² + ∂²u/∂y² = 0` on a unit square `0 < x < 1, 0 < y < 1`.
*   **Boundary Conditions**: `u(x,1)=2`, `u(1,y)=2`, `u(x,0)=1`, `u(0,y)=0`.
*   **Method**: **Jacobi Iteration**
    *   **Logic**: The Laplace equation is discretized using a 5-point central finite difference stencil, yielding an iterative update rule `u_i,j^(k+1) = (1/4) * (u_{i+1,j}^(k) + u_{i-1,j}^(k) + u_{i,j+1}^(k) + u_{i,j-1}^(k))`.
    *   **Implementation**: A Python function `solve_laplace_jacobi` is implemented to apply the Jacobi iteration until a specified convergence tolerance is met.
    *   **Grid Resolutions**: Solutions are computed and visualized for two different grid sizes (`Δx = Δy = 1/3` and `Δx = Δy = 1/6`).
    *   **Visualization**: `Matplotlib` is used to create 3D surface plots and 2D contour plots of the converged solution, illustrating the potential distribution.

### 4. Gauss-Seidel Convergence Proof

This final section provides a theoretical proof for the convergence of the Gauss-Seidel iterative method.
*   **Purpose**: To demonstrate that the Gauss-Seidel iteration for solving `Ax = b` converges for any initial guess if the matrix `A` is strictly diagonally dominant.
*   **Logic**: The proof outlines the decomposition of matrix `A` into `D + L + U`, defines the Gauss-Seidel iteration matrix `G_GS = -(D + L)⁻¹ U`, and states the convergence criterion `ρ(G_GS) < 1` (spectral radius less than 1). It explains how strict diagonal dominance ensures this condition, guaranteeing the convergence of the error vector to zero.

## Project Requirements

To run the Jupyter notebooks in this repository, you need the following Python libraries installed:
*   `numpy`
*   `matplotlib`
*   `sympy`
*   `jupyter`

## Installation

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/Hazem-74/Random-Problems.git
    ```
2.  **Navigate to the Directory**:
    ```bash
    cd Random-Problems/Math
    ```
3.  **Install Dependencies**:
    It's recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install numpy matplotlib sympy jupyter
    ```

## How to Run the Notebooks

1.  **Start Jupyter Notebook**:
    After installing the requirements and navigating to the `Math` directory, launch Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
2.  **Open `Numerical.ipynb`**:
    Your web browser will open, displaying the Jupyter interface. Click on `Numerical.ipynb` to open the notebook.
3.  **Execute Cells**:
    You can run the code cells sequentially by pressing `Shift + Enter` or by using the "Run" button in the toolbar.

## License

This project is open-source and available under the [MIT License](LICENSE).

---