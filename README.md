# Hazem-74's Random Problems Repository

This repository, `/home/salastro/Documents/Hazem's Bible/Random Problems`, contains a collection of Jupyter notebooks exploring various topics in physics, mathematics, and data science. Each notebook addresses a specific problem or concept, often involving symbolic computation, numerical methods, simulations, and data analysis.

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Notebook Summaries](#notebook-summaries)
  - [Lane-Emdden.ipynb](#lane-emddenipynb)
  - [Eigenvalues.ipynb](#eigenvaluesipynb)
  - [Teams.ipynb](#teamsipynb)
  - [two_opt_heuristic.ipynb](#two_opt_heuristicipynb)
  - [OOP.ipynb](#oopipynb)
  - [Simplification.ipynb](#simplificationipynb)
  - [Mustafa.ipynb](#mustafaipynb)
  - [nearest_neighbor_heuristic.ipynb](#nearest_neighbor_heuristicipynb)
  - [Christofides_algorithm.ipynb](#christofides_algorithmipynb)
  - [Simplifications.ipynb](#simplificationsipynb)
  - [Quantum_OTP.ipynb](#quantum_otpipynb)
  - [Linalg.ipynb](#linalgipynb)
  - [Induced Metric.ipynb](#induced-metricipynb)
  - [Assignemt.ipynb](#assignemtipynb)
  - [Problem_2.52.ipynb](#problem_252ipynb)
  - [delta-f(x).ipynb](#delta-fxipynb)
  - [Recurrency.ipynb](#recurrencyipynb)
  - [mod5_final_project.ipynb](#mod5_final_projectipynb)
  - [n = 3 polytropic model.ipynb](#n--3-polytropic-modelipynb)
  - [Participation 2.ipynb](#participation-2ipynb)
  - [Simplifications2.ipynb](#simplifications2ipynb)
  - [Random Walk.ipynb](#random-walkipynb)
  - [Question7.ipynb](#question7ipynb)
  - [MoMo.ipynb](#momoipynb)
  - [polytropic equation.ipynb](#polytropic-equationipynb)
  - [Harmonic_Osci.ipynb](#harmonic_osciipynb)
  - [Untitled.ipynb](#untitledipynb)
  - [spherical wave and Fresnel Equ.ipynb](#spherical-wave-and-fresnel-equipynb)
  - [Stat.ipynb](#statipynb)
  - [ChurnPrediction.ipynb](#churnpredictionipynb)
  - [PSI.ipynb](#psiipynb)

---

## Project Overview

This repository serves as a personal collection of computational and analytical problems. It includes explorations into astrophysics, quantum mechanics, statistical mechanics, linear algebra, algorithms for combinatorial optimization, and data science challenges. Each notebook is self-contained, presenting the problem, methods, logic, and results.

## Repository Structure

The repository is structured as a flat directory of Jupyter Notebook files (`.ipynb`).

```
Random Problems/
├── Lane-Emdden.ipynb
├── Eigenvalues.ipynb
├── Teams.ipynb
├── two_opt_heuristic.ipynb
├── OOP.ipynb
├── Simplification.ipynb
├── Mustafa.ipynb
├── nearest_neighbor_heuristic.ipynb
├── Christofides_algorithm.ipynb
├── Simplifications.ipynb
├── Quantum_OTP.ipynb
├── Linalg.ipynb
├── Induced Metric.ipynb
├── Assignemt.ipynb
├── Problem_2.52.ipynb
├── delta-f(x).ipynb
├── Recurrency.ipynb
├── mod5_final_project.ipynb
├── n = 3 polytropic model.ipynb
├── Participation 2.ipynb
├── Simplifications2.ipynb
├── Random Walk.ipynb
├── Question7.ipynb
├── MoMo.ipynb
├── polytropic equation.ipynb
├── Harmonic_Osci.ipynb
├── Untitled.ipynb
├── spherical wave and Fresnel Equ.ipynb
├── Stat.ipynb
├── ChurnPrediction.ipynb
├── PSI.ipynb
└── ... (other files as added)
```

## Requirements

To run these notebooks, you will need to have Python 3 installed, along with several scientific computing and data science libraries. The primary libraries used across the notebooks include:

*   `Jupyter Notebook` or `JupyterLab`
*   `Python 3.x`
*   `numpy`
*   `scipy`
*   `matplotlib`
*   `sympy`
*   `qiskit` (and `qiskit-aer`)
*   `pandas`
*   `sqlite3`
*   `networkx`
*   `seaborn`
*   `scikit-learn`
*   `IPython.display`

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Hazem-74/Random-Problems.git
    cd Random-Problems
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    ```bash
    pip install numpy scipy matplotlib sympy qiskit qiskit-aer pandas networkx seaborn scikit-learn jupyter
    ```
    Some notebooks might require additional packages. If you encounter an `ImportError`, simply `pip install` the missing library.

## Usage

1.  **Start Jupyter Notebook or JupyterLab:**
    ```bash
    jupyter notebook
    # or
    jupyter lab
    ```

2.  **Navigate to the desired notebook:** In your web browser, select the `.ipynb` file you wish to explore.

3.  **Run the cells:** Execute the code cells sequentially to see the derivations, computations, and visualizations.

## Notebook Summaries

### Lane-Emdden.ipynb
*   **Purpose:** Numerically solve the Lane-Emden equation for polytropic stars and derive internal stellar structure (density, pressure, temperature, luminosity profiles).
*   **Methods:** Finite difference method for solving a second-order ordinary differential equation, linear interpolation to find boundary conditions, trapezoidal rule for numerical integration, and standard astrophysical formulas.
*   **Main Logic:** The notebook discretizes the Lane-Emden equation, iteratively steps outwards from the stellar core, interpolates to find the surface ($\theta=0$), and then uses derived constants and the solution to generate and plot radial profiles of density, pressure, temperature, and luminosity, comparing with standard solar models.

### Eigenvalues.ipynb
*   **Purpose:** Symbolically compute the eigenvalues and eigenvectors (eigenstates) of a 2x2 Hermitian matrix.
*   **Methods:** `sympy` library for symbolic mathematics, `sp.Matrix` to define the matrix, `eigenvals()` and `eigenvects()` methods.
*   **Main Logic:** It defines a general 2x2 Hermitian matrix using symbolic variables, then uses SymPy's built-in functions to find its characteristic polynomial roots (eigenvalues) and the corresponding basis vectors (eigenstates), simplifying the results for clarity.

### Teams.ipynb
*   **Purpose:** Develop an algorithm to assign 8 teams into 16 unique pairs across two shifts, minimizing overlap between shifts, and visualize the assignments using graph theory.
*   **Methods:** `itertools.combinations` to generate all possible pairs, `random.shuffle` for randomized selection, `collections.defaultdict` for tracking team counts, `networkx` for graph creation, and `matplotlib` for visualization.
*   **Main Logic:** It generates all possible team pairings. It then constructs two shifts by randomly selecting 16 non-repeating pairs for each shift while ensuring a maximum number of participations per team and iteratively searching for shift combinations that minimize common pairs. Finally, it maps pairs to "places" and visualizes the team matchups as graphs for each shift.

### two_opt_heuristic.ipynb
*   **Purpose:** Implement the 2-opt local search heuristic algorithm to find an improved (shorter) tour for the Traveling Salesperson Problem (TSP).
*   **Methods:** `networkx` to represent the complete graph of cities, Euclidean distance calculation for edge weights, and an iterative 2-opt swap logic.
*   **Main Logic:** The algorithm starts with an arbitrary initial tour. It then repeatedly identifies two non-adjacent edges in the tour, checks if swapping them for two new edges (which effectively reverses a segment of the tour) reduces the total tour length. This process continues until no further improvements can be found, thereby converging to a locally optimal tour. The tour is visualized at each improvement step.

### OOP.ipynb
*   **Purpose:** This notebook is currently empty and likely serves as a placeholder for future Object-Oriented Programming (OOP) exercises or demonstrations.
*   **Methods:** N/A (Empty Notebook)
*   **Main Logic:** N/A (Empty Notebook)

### Simplification.ipynb
*   **Purpose:** Solve and visualize transcendental equations from quantum mechanics (finite square well), and derive reflection/transmission coefficients for potential barriers/steps. It also explores a statistical mechanics function.
*   **Methods:** `sympy` for symbolic algebra and equation solving, `numpy` and `matplotlib` for numerical plotting and finding intersections graphically.
*   **Main Logic:** For the finite square well, it derives symbolic equations for even/odd solutions and then plots these equations numerically to find their intersections, which correspond to allowed energy levels. For potential barriers/steps, it sets up boundary conditions, solves for symbolic reflection/transmission amplitudes, and calculates coefficients. The notebook also includes symbolic limits and plots for a statistical mechanics function `f(T) = sech^2(E/kBT) / T`.

### Mustafa.ipynb
*   **Purpose:** Provide a detailed theoretical overview of $f(T)$ teleparallel gravity, discussing its field equations, power series expansion, cosmological applications, and associated challenges.
*   **Methods:** Primarily theoretical exposition with mathematical derivations presented in Markdown. No executable code related to the physics content, but it uses Jupyter's rich text features for scientific writing.
*   **Main Logic:** It outlines the core field equations of $f(T)$ gravity, including the energy density and dynamical equations, and derives a key differential equation for the torsion scalar $T(t)$. The notebook then explores power series expansions of $f(T)$, discussing the implications of higher-order terms for early universe inflation, late-time acceleration, and bounce cosmologies, while also highlighting theoretical challenges like instabilities and naturalness.

### nearest_neighbor_heuristic.ipynb
*   **Purpose:** Implement the Nearest Neighbor heuristic algorithm for the Traveling Salesperson Problem (TSP).
*   **Methods:** `networkx` to represent the graph of cities, Euclidean distance calculation for edge weights, and a greedy selection approach.
*   **Main Logic:** The algorithm starts at an arbitrary city. From the current city, it iteratively selects the unvisited city that is closest (nearest neighbor) and adds it to the tour. This process continues until all cities have been visited, forming a complete tour. The final tour is then visualized.

### Christofides_algorithm.ipynb
*   **Purpose:** Implement Christofides' 1.5-approximation algorithm for the Traveling Salesperson Problem (TSP), which guarantees a tour at most 1.5 times the optimal length.
*   **Methods:** `networkx` for graph-theoretic operations: Minimum Spanning Tree (`nx.minimum_spanning_tree`), identifying odd-degree nodes, Minimum-Cost Perfect Matching (`nx.max_weight_matching`), and finding an Eulerian circuit (`nx.eulerian_circuit`). Euclidean distance calculation for edge weights.
*   **Main Logic:** The algorithm constructs an MST, finds odd-degree nodes within the MST, and then adds a minimum-cost perfect matching on these odd-degree nodes. This creates an Eulerian multigraph. An Eulerian circuit is then extracted from this multigraph, and "shortcuts" are taken (skipping already visited nodes) to convert it into a valid Hamiltonian TSP tour.

### Simplifications.ipynb
*   **Purpose:** Demonstrate symbolic vector algebra related to vector rotations (Rodrigues' formula) and the properties of the Levi-Civita tensor using SymPy.
*   **Methods:** `sympy` for symbolic mathematics, `sympy.Matrix` for vector and matrix operations, `sympy.vector` (implicitly for tensor representation) and `numpy` for tensor manipulation.
*   **Main Logic:** The notebook first defines and demonstrates the Levi-Civita tensor and its transpose to show its anti-symmetric property. It then implements Rodrigues' rotation formula, applying it to a generic vector `r` rotated around a unit axis `n_hat` by an angle `phi`. It constructs the corresponding rotation matrix `S` and verifies its orthogonality by computing `S * S.T = I`.

### Quantum_OTP.ipynb
*   **Purpose:** Implement a complete Quantum One-Time Pad (QOTP) protocol for encrypting and decrypting a binary message using Qiskit.
*   **Methods:** `qiskit` for quantum circuit construction and simulation, `QuantumRegister`, `ClassicalRegister`, `QuantumCircuit`, `AerSimulator` for running quantum circuits, `random.randrange` for generating classical keys.
*   **Main Logic:** The protocol involves encoding a binary message onto qubits using `X` gates. Encryption is performed by applying random `X` (bit-flip) and `Z` (phase-flip) gates based on a pre-shared classical key. Decryption reverses this process by applying the same `Z` and `X` gates with the same key. The message is measured after encryption and decryption, and the final decrypted message is verified against the original.

### Linalg.ipynb
*   **Purpose:** Transform quadratic forms into their canonical (diagonalized) form using eigenvalue decomposition, and provide a generalized function for this transformation.
*   **Methods:** `sympy` for symbolic linear algebra, `eigenvects()` method to find eigenvalues and eigenvectors, matrix multiplication, and symbolic substitution.
*   **Main Logic:** It starts with a concrete example, extracting the symmetric matrix `V` from a given quadratic form. It then computes the eigenvectors of `V`, constructs a transformation matrix from these normalized eigenvectors, and applies a change of variables to diagonalize the quadratic form. A general function `Transforming` is then created to automate this process for various quadratic expressions.

### Induced Metric.ipynb
*   **Purpose:** Calculate the induced metric, its determinant, and the area element for a parameterized surface (specifically, a hyperboloid of one sheet) embedded in Minkowski spacetime, and visualize the surface and area properties.
*   **Methods:** `sympy` for symbolic calculus and matrix operations, `numpy` and `matplotlib.pyplot` (with `mpl_toolkits.mplot3d`) for numerical computation and 3D visualization.
*   **Main Logic:** It defines the parametric equations of the hyperboloid. It then computes the partial derivatives of the embedding coordinates with respect to the surface parameters (`u`, `v`) to form the Jacobian. Using the ambient Minkowski metric, it calculates the induced metric `h_ab = J^T g J`, its determinant, and the area element `dA = sqrt(|det(h_ab)|) du dv`. The notebook also visualizes the hyperboloid and plots the numerical behavior of the area element and total area.

### Assignemt.ipynb
*   **Purpose:** Solve several quantum computing problems using Qiskit, focusing on circuit construction, state evolution, and measurement outcomes.
*   **Methods:** `qiskit` for quantum circuits (`QuantumCircuit`, `QuantumRegister`, `ClassicalRegister`), `Statevector` for state simulation, `AerSimulator` for running circuits, `numpy` for random state generation, `matplotlib.pyplot` for result histograms.
*   **Main Logic:** The notebook contains solutions to three quantum computing problems (Question 1, 3, and 4). Question 1 involves building a specific 3-qubit circuit with Hadamard and CNOT gates and observing its state and measurement counts. Question 3 defines a function to generate random 1-qubit states and then uses such a state in a 2-qubit circuit. Question 4 expands on this by using two random 1-qubit states in a 2-qubit circuit with entanglement operations.

### Problem_2.52.ipynb
*   **Purpose:** Analyze the quantum mechanics of a particle in a `sech^2(ax)` potential well, specifically finding the ground state energy, normalizing its wavefunction, and deriving reflection/transmission coefficients for scattering states.
*   **Methods:** `sympy` for symbolic calculus (differentiation, integration, solving equations), `numpy` and `matplotlib.pyplot` for numerical plotting and visualization.
*   **Main Logic:** The notebook first visualizes the `sech^2(ax)` potential. For bound states, it assumes a trial ground state wavefunction proportional to `sech(ax)`, substitutes it into the Schrödinger equation to find the corresponding energy eigenvalue, and then normalizes the wavefunction. For scattering states, it uses a known scattering solution to derive the reflection and transmission coefficients and analyzes their asymptotic behavior.

### delta-f(x).ipynb
*   **Purpose:** Demonstrate symbolic integration of products of Dirac Delta functions using SymPy.
*   **Methods:** `sympy.DiracDelta` for representing the Dirac delta function, `sympy.integrate` for symbolic integration.
*   **Main Logic:** The notebook calculates two specific integrals: the integral of a product of two shifted Dirac delta functions, $\int \delta(x-\alpha)\delta(x-\beta) dx$, and the integral of a squared Dirac delta function, $\int \delta(x-\alpha)^2 dx$, to illustrate the symbolic handling of these distributions.

### Recurrency.ipynb
*   **Purpose:** Perform symbolic differentiation of a partition function and a related series in statistical mechanics with respect to temperature.
*   **Methods:** `sympy` for symbolic operations (`symbols`, `Function`, `diff`, `summation`, `exp`, `simplify`).
*   **Main Logic:** It defines a partition function `Z_expr` and a generalized sum `g_n` involving `Z_expr` and a summation over `l`. The primary goal is to symbolically compute the derivative of `g_n` with respect to temperature `T`. Due to the complexity of the symbolic summation, the notebook also explores iterative differentiation of `Z_expr`.

### mod5_final_project.ipynb
*   **Purpose:** A data science project to load, explore, and query three Chicago datasets (Census, Public Schools, Crime) using Pandas and an SQLite database.
*   **Methods:** `pandas` for data loading and manipulation, `sqlite3` for database connection and table creation, SQL queries for data retrieval and analysis, and basic visualization with `matplotlib` and `seaborn`.
*   **Main Logic:** The notebook downloads CSV datasets into Pandas DataFrames, then uploads these DataFrames to three tables within an `SQLite` database (`FinalDB.db`). It then executes a series of SQL queries to answer specific questions about the data, such as counting total crimes, finding areas with low income, identifying crime patterns, and calculating statistics on school safety scores and poverty levels.

### n = 3 polytropic model.ipynb
*   **Purpose:** Analyze a polytropic stellar model with index `n=3`, focusing on the numerical solution of the Lane-Emden equation to derive temperature and density profiles, and calculate total nuclear energy generation.
*   **Methods:** `scipy.integrate.odeint` for solving ordinary differential equations (Lane-Emden), `numpy` for numerical array operations, `matplotlib.pyplot` for plotting, and `scipy.integrate.quad` for numerical integration.
*   **Main Logic:** It defines and numerically solves the Lane-Emden equation for `n=3`. Using central temperature and density values, it then generates radial profiles for temperature and density. The notebook also calculates the total luminosity by numerically integrating a given energy generation rate formula throughout the stellar interior and visualizes the linear and logarithmic profiles.

### Participation 2.ipynb
*   **Purpose:** Illustrate the behavior of electron degeneracy pressure ($P$) as a function of the density ratio ($\rho/\rho_e$) across non-relativistic and ultra-relativistic regimes.
*   **Methods:** `numpy` for numerical calculations (`np.linspace`, `np.logspace`, `np.sqrt`, `np.arcsinh`), and `matplotlib.pyplot` for log-log plotting.
*   **Main Logic:** The notebook defines fundamental physical constants and a specific function `f(x)` derived from the electron degeneracy pressure formula. It calculates $P$ and the density ratio based on a momentum factor `x`. A log-log plot of $P$ versus $\rho/\rho_e$ is generated, with superimposed lines indicating the characteristic slopes for the non-relativistic ($P \propto \rho^{5/3}$) and ultra-relativistic ($P \propto \rho^{4/3}$) limits, highlighting the transition between these regimes.

### Simplifications2.ipynb
*   **Purpose:** Perform symbolic vector calculus and matrix operations related to electromagnetic fields and a complex vector `psi` constructed from `E` and `B` fields, using SymPy.
*   **Methods:** `sympy` for symbolic mathematics, `sympy.Matrix` for vector and matrix operations, `sympy.I` for the imaginary unit, and `sympy.diff` for symbolic differentiation.
*   **Main Logic:** The notebook defines symbolic electric (`E`) and magnetic (`B`) field components and then constructs a complex vector `psi` as a combination of these fields. It defines specific Pauli-like matrices (`M_x`, `M_y`, `M_z`) and a symbolic gradient operator. The main computation involves forming expressions like `(M . ∇)ψ` or similar combinations involving time derivatives, exploring the mathematical structure of field transformations.

### Random Walk.ipynb
*   **Purpose:** Provide both analytical derivation and numerical simulation of symmetric random walks in 1D and 3D, specifically modeling photon diffusion within a sphere.
*   **Methods:** Theoretical derivation using induction for 1D, `numpy` for numerical simulation of 3D random walks (generating random spherical angles), and `matplotlib.pyplot` (with `mpl_toolkits.mplot3d`) for 3D path visualization.
*   **Main Logic:** The notebook first derives that the mean squared displacement `<x_n^2>` for an `n`-step 1D random walk is `nλ^2`, leading to `N_out = d^2/λ^2` to reach distance `d`. It then extends this derivation to 3D and explains why the result remains the same. The numerical part simulates `Nphotons` performing a 3D random walk, calculating the average number of steps `N_out` to escape a sphere of radius `d`, and visualizes a single photon's trajectory.

### Question7.ipynb
*   **Purpose:** Calculate the allowed energy levels for a particle in a finite square well and plot the corresponding wavefunctions using numerical root-finding techniques.
*   **Methods:** `numpy` for numerical arrays, `scipy.optimize.root_scalar` for solving transcendental equations, `matplotlib.pyplot` for plotting.
*   **Main Logic:** The notebook defines dimensionless transcendental equations that arise from the boundary conditions of the finite square well for both even and odd solutions. It uses a numerical root-finding method to find the roots (which correspond to quantized energy levels). These energies are then converted back to physical units, and the corresponding cosine (even) and sine (odd) wavefunctions are plotted within the well. The process is applied to two different well configurations (Part A and Part B).

### MoMo.ipynb
*   **Purpose:** Symbolically verify if a specific relation involving a complex matrix `W` and a skew-symmetric matrix `o` (i.e., `W.T * o * W = o`) is equivalent to the unitarity condition of `W` (i.e., `W.H * W = I`). This explores properties of pseudo-unitary or symplectic transformations in a specific context.
*   **Methods:** `sympy` for symbolic matrix operations (`sp.Matrix`, `sp.conjugate`, `sp.H` for Hermitian transpose, `sp.T` for transpose, `sp.eye`), `sp.simplify`, `sp.Eq`.
*   **Main Logic:** The notebook defines `W` with complex symbolic entries and `o`. It computes the two conditions: `M = W.T * o * W - o = 0` and `U_w = W.H * W - I = 0`. It then extracts the individual equations from both matrix conditions and uses a custom `are_equivalent` function to check if the set of equations from the first condition is symbolically identical (up to a sign) to the set of equations from the unitarity condition.

### polytropic equation.ipynb
*   **Purpose:** Prove and numerically model the pressure profile of an atmosphere under a polytropic equation of state ($P = K \rho^{5/3}$), applying the model to Jupiter's atmosphere.
*   **Methods:** Theoretical derivation combining the polytropic equation with hydrostatic equilibrium, `numpy` for numerical calculations, and `matplotlib.pyplot` for plotting.
*   **Main Logic:** The notebook first provides a step-by-step derivation of the pressure profile $P(z)$ as a function of height $z$ and the maximum altitude $z_{max}$ where pressure vanishes, assuming a polytropic index $n=1.5$. It then estimates the polytropic constant $K$ for Jupiter using its mass, radius, and approximate surface pressure. Finally, it numerically calculates and plots the pressure profile $P(z)$ for Jupiter's atmosphere up to $z_{max}$.

### Harmonic_Osci.ipynb
*   **Purpose:** Explore the quantum harmonic oscillator (QHO) problem by deriving its symbolic Schrödinger equation, energy eigenvalues, and eigenfunctions, and then visualizing the eigenfunctions and probability densities numerically.
*   **Methods:** `sympy` for symbolic representation of the Schrödinger equation, eigenvalues, eigenfunctions, and Hermite polynomials. `numpy` and `matplotlib.pyplot` for numerical plotting, utilizing `scipy.special.hermite` for Hermite polynomial evaluation.
*   **Main Logic:** The notebook begins by stating the symbolic time-independent Schrödinger equation for the QHO. It then presents and displays the normalized energy eigenfunctions (involving Hermite polynomials) and corresponding energy eigenvalues. The numerical part defines functions to calculate the eigenfunctions and probability densities for given quantum numbers $n$ and plots them, illustrating the characteristic shapes of QHO wavefunctions and how the probability density approaches classical behavior for high $n$.

### Untitled.ipynb
*   **Purpose:** An exploratory notebook attempting to symbolically solve a differential equation for `f(T)` arising in $f(T)$ teleparallel gravity, based on a power series expansion for `T(t)`.
*   **Methods:** `sympy` for symbolic calculus (`Function`, `Derivative`, `sqrt`, `Rational`, `dsolve`), and `numpy` and `matplotlib` for optional numerical plotting.
*   **Main Logic:** The notebook sets up the fundamental equations from $f(T)$ gravity relating the function $f(T)$ to the torsion scalar $T$, Hubble parameter $H$, and other cosmological parameters. It then attempts to find a symbolic solution for $f(T)$ by assuming $T(t)$ has a power series expansion in time $t$, and expressing $H$ and $\dot{T}$ in terms of $T$. The notebook builds a symbolic ordinary differential equation (ODE) for $f(T)$ and tries to solve it using `sympy.dsolve`.

### spherical wave and Fresnel Equ.ipynb
*   **Purpose:** Analyze a given spherical electromagnetic wave by calculating its curl and divergence using symbolic vector calculus, and demonstrate Fresnel equations for reflection and transmission at an interface.
*   **Methods:** `sympy` for symbolic calculus (`symbols`, `sin`, `cos`, `Function`, `simplify`, `integrate`, `diff`, `sqrt`), `sympy.vector` for vector operations (`CoordSys3D`, `curl`, `dot`, `express`, `Del`), `numpy` and `matplotlib.pyplot` for numerical plotting.
*   **Main Logic:** For the spherical wave, it defines the electric field `E` in spherical coordinates. It then symbolically computes `∇ × E` to find the magnetic field `B`, followed by `∇ ⋅ B` and `∇ × B`. It also calculates the Poynting vector `S = B × E`. For Fresnel equations, it numerically plots the reflected ($E_R$) and transmitted ($E_T$) electric field amplitudes as functions of the angle of incidence, for a given refractive index ratio `beta`, and symbolically verifies the energy conservation relation $R+T=I$.

### Stat.ipynb
*   **Purpose:** Illustrate a statistical mechanics function involving the hyperbolic secant and calculate energy levels and their degeneracies for a 2D particle in a box.
*   **Methods:** `numpy` and `matplotlib.pyplot` for numerical function plotting, `pandas` for structured data display.
*   **Main Logic:** The notebook first plots the function $f(T) = \frac{1}{T} \text{sech}^2\left(\frac{E}{k_B T}\right)$ for both positive and negative temperatures. Subsequently, for a 2D particle in a box (Part i), it defines functions to calculate energy levels ($E = \epsilon_0 (n_x^2 + n_y^2)$) and their corresponding degeneracies (number of $(n_x, n_y)$ states for a given energy) up to a maximum energy, presenting the results in a Pandas DataFrame.

### ChurnPrediction.ipynb
*   **Purpose:** Develop a machine learning model to predict customer churn for a video streaming service using a provided dataset, as part of a data science challenge.
*   **Methods:** `pandas` for data manipulation, `numpy`, `matplotlib`, `seaborn` for data exploration and visualization, `scikit-learn` for machine learning (preprocessing, model training, prediction, evaluation). Specifically uses `StandardScaler`, `OneHotEncoder`, `ColumnTransformer`, `Pipeline`, and `LogisticRegression`.
*   **Main Logic:** The notebook loads training and testing datasets. It performs data exploration (e.g., handling duplicates and missing values, visualizing distributions and correlations). It then preprocesses the data by scaling numerical features and one-hot encoding categorical features. A `LogisticRegression` model is trained within a `Pipeline`, validated using `roc_auc_score`, and finally used to predict churn probabilities on the test set, formatted for submission.

### PSI.ipynb
*   **Purpose:** Verify a specific functional form `f(x,t) = exp(i*theta(x,t))` for a free particle's wavefunction by substituting it into related differential equations, demonstrating its properties.
*   **Methods:** `sympy` for symbolic calculus (`symbols`, `exp`, `I` for imaginary unit, `diff`, `simplify`).
*   **Main Logic:** The notebook defines the phase factor `theta(x,t)` for a free particle (in terms of momentum `p` and kinetic energy). It then defines the wavefunction `f(x,t)` as `exp(i*theta)`. The core logic involves calculating the first and second spatial derivatives (`f_x`, `f_xx`) and the time derivative (`f_t`) of `f`. It then defines two expressions `A` and `B` which correspond to parts of the Schrödinger equation and the momentum operator, simplifying them to show that `A` reduces to zero (verifying `f` as a solution to the free particle Schrödinger equation) and `B` relates to the momentum.

---