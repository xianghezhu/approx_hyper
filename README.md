# approx_hyper
Code for Approximating Distributions on Vertex Degrees in Random Hypergraphs via Stein’s Method.


## Simulation

- [multipois_p.py](simulation/multipois_p.py) and [multipois_n_sparse.py](simulation/multipois_n_sparse.py): compare joint degree distributions with multivariate Poisson variables, varying edge probabilities or hypergraph size in the sparse regime.
- [multinorm_p.py](simulation/multinorm_p.py) and [multinorm_n_dense.py](simulation/multinorm_n_dense.py): compare standardized degrees with a multivariate normal distribution, varying edge probabilities or hypergraph size in the dense regime.
- [application_n_sparse.py](simulation/application_n_sparse.py): studies probability errors for counts of vertices with degree zero or one.
- [application2_run.py](simulation/application2_run.py): compares Gaussian approximations and rejection rates for maximum degrees under an ER hypergraph, a hypergraph stochastic block model, and a simplicial complex. The sampling functions are in [application2_sampling.py](simulation/application2_sampling.py).

## Empirical analysis

- [madstat.py](empirical/madstat.py): analyzes the MADStat coauthorship data from 1975–2015, using all hyperedges of size at least two and the subset of size at least six.
- [primary_school.py](empirical/primary_school.py): analyzes primary school contact data for the whole school and class 5A.
- [madstat_ci.py](empirical/madstat_ci.py) and [primary_school_ci.py](empirical/primary_school_ci.py): compute simulation means and central 95% intervals under fitted random hypergraph models.

Input data are included in `empirical/data/`. The file [prepare_madstat.R](empirical/data/madstat/prepare_madstat.R) exports MADStat text files from `AuthorPaperInfo.RData`. 
