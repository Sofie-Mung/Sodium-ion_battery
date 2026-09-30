# inputs/

* `Li6PS5Cl.cif` — **user-provided host CIF (required)**. Preferred: an experimental
  refinement with partial occupancies (Li 48h ~0.5, S/Cl mixed on 4a/4c), space group F-43m.
  The pipeline also accepts an ordered CIF whose Li orbit under the framework symmetry has
  48 sites per unit cell; anything else is rejected with an explicit message.
* `exp_reference.csv` — optional experimental reference values (σ at RT in mS/cm, Ea in eV,
  lattice a in Å). Empty cells are allowed; the code then skips absolute comparisons.
