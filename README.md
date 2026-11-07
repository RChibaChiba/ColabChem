# ColabChem

**English** | [日本語](README.ja.md)

[Homepage (ColabChem)](https://rchibachiba.github.io/ColabChem/en/) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24035698.svg)](https://doi.org/10.5281/zenodo.24035698)

| Notebook | What it does | Open |
|---|---|---|
| ColabUMA Opt | Geometry optimization → DFT single point | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RChibaChiba/ColabChem/blob/main/ColabUMA_Opt.ipynb) |
| ColabUMA Scan | Relaxed scan (bond length, angle, dihedral) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RChibaChiba/ColabChem/blob/main/ColabUMA_Scan.ipynb) |
| ColabUMA TS | Transition state search (DMF → Sella → DFT single point) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RChibaChiba/ColabChem/blob/main/ColabUMA_TS.ipynb) |
| ColabUMA Workflow | Optimization → scan → transition state in one session | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RChibaChiba/ColabChem/blob/main/ColabUMA_Workflow.ipynb) |
| ColabDFT SPE | DFT single point only (no UMA, no token needed) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RChibaChiba/ColabChem/blob/main/ColabDFT_SPE.ipynb) |
| ColabDFT ESP | DFT single point and 3D view of the electrostatic potential (ESP) (no UMA, no token needed) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RChibaChiba/ColabChem/blob/main/ColabDFT_ESP.ipynb) |
| ColabDFT MO | DFT single point and 3D view of molecular orbitals such as HOMO and LUMO (no UMA, no token needed) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RChibaChiba/ColabChem/blob/main/ColabDFT_MO.ipynb) |

Google Colab notebooks for geometry optimization, relaxed scans and transition state searches of molecules with Meta's machine learning potential **UMA**. Open a notebook in Colab and choose "Run all". You only need a structure (xyz or Gaussian gjf), the charge and the spin multiplicity.

The notebooks' forms, messages and logs are in English. Error messages also show a Japanese translation. The explanation at the top of each notebook is in Japanese; this README covers the same content in English.

<!-- TODO: add a screenshot (e.g. docs/screenshot.png) -->

## Features

- Geometry optimization from an xyz file or a Gaussian input file (.gjf / .com) (LBFGS / BFGS / FIRE)
- The optimization path is also written as a Gaussian log, so GaussView shows each step's structure and the energy profile
- Vibrational analysis to check for imaginary frequencies (also written in the format of Gaussian's freq output, so GaussView can animate the modes)
- DFT single point at the optimized structure (GPU4PySCF)
- 3D view, energies (eV and Hartree), a convergence plot, and a ZIP of all results to download

- Relaxed scans of a bond length, angle or dihedral, with optimizations before and after the scan ([ColabUMA Scan](#colabuma-scan))
- Transition state search from a reactant and a product (DMF → Sella), imaginary frequency check, a simple IRC check, and DFT single points ([ColabUMA TS](#colabuma-ts))
- Optimization → scan → transition state in a row, with a single install and model load ([ColabUMA Workflow](#colabuma-workflow))
- DFT single point of your own structure as it is ([ColabDFT SPE](#colabdft-spe))
- ESP of your own structure, colored on an electron density isosurface ([ColabDFT ESP](#colabdft-esp))
- Molecular orbitals such as HOMO and LUMO of your own structure, shown as isosurfaces ([ColabDFT MO](#colabdft-mo))

Not covered: a real IRC calculation (the simple IRC check in the TS notebook only displaces along the imaginary mode and optimizes), and periodic systems (crystals, surfaces).

## Before you start (first time only)

The model weights, [facebook/UMA](https://huggingface.co/facebook/UMA), require an access request.

1. Create a [Hugging Face](https://huggingface.co/join) account
2. Request access on the [facebook/UMA](https://huggingface.co/facebook/UMA) page and accept the FAIR Chemistry License v1 (wait for approval)
3. Create a token with **Read** permission under [Access Tokens](https://huggingface.co/settings/tokens)
4. Open a notebook in Colab. In the 🔑 (Secrets) panel of the left sidebar, add a secret named `HF_TOKEN` with the token as its value, and turn on "Notebook access"

If you don't add the secret, the notebook shows a token input box before the file selection. The token is not saved in the notebook.

## How to use

1. Open a notebook from its "Open in Colab" badge above
2. Choose **Runtime → Change runtime type** and select **T4 GPU**
3. First time only: add `HF_TOKEN` to the secrets and allow notebook access
4. In "1. Input form", set the output name, charge, spin multiplicity and convergence criterion
5. Choose **Runtime → Run all**
6. A file selection button appears in "2. File selection and checks". Choose an xyz or gjf file. From here on, the install and the calculation run unattended
7. Check the 3D view, the energies and the convergence plot
8. Download the ZIP of the results

For later runs, repeat steps 4–8. The install and the model load are skipped automatically. If you press "Cancel upload" in the file selection, the previous file is used again.

### Input form

| Item | Default | Description |
|---|---|---|
| Output name | `result` | Prefix of the output file names |
| Charge | 0 | Total charge of the molecule |
| Spin multiplicity | 1 | 2S+1. An error if it does not match the parity of the electron count |
| Model | uma-s-1p1 | uma-s-1p1 / uma-s-1p2 / uma-m-1p1 |
| Task | omol | Only omol (molecules) in v1 |
| Optimizer | LBFGS | LBFGS / BFGS / FIRE (ASE) |
| Convergence | loose (0.05) | fmax (eV/Å). loose 0.05 / normal 0.02 (about Gaussian's default) / tight 0.001 (about Gaussian's opt=tight) / custom |
| Custom fmax | 0.05 | Used when "custom" is chosen for the convergence |
| Max steps | 500 | |
| Vibrational analysis | Off | If on, checks for imaginary frequencies after the optimization |
| DFT single point | Off | Computes the DFT energy at the optimized structure (GPU4PySCF, density fitting). The structure stays the UMA one; it is not reoptimized with DFT |
| Functional / basis set | B3LYP-D3(BJ) / def2-SVP | The default favors speed. ωB97M-V/def2-TZVPD is the level of theory of UMA's training data (OMol25), but it is slow on a T4 |
| DFT time limit | 60 min | Time limit for the DFT single points (total over all structures). When it is exceeded, DFT is skipped and the UMA results are saved. 0 means no limit ([DFT memory and time](#dft-memory-and-time)) |

An xyz file holds one structure. If it holds several, only the first is used. Examples: [examples/example.xyz](examples/example.xyz) and [examples/example.gjf](examples/example.gjf).

From a Gaussian input file (.gjf / .com / .gjc), **only the structure** is read. The route section (method, basis set, etc.) is ignored, and the charge and multiplicity come from the form (a note is shown if they differ from the gjf). Besides Cartesian coordinates, the reader handles the atom-type lines written by GaussView (`C-CA--0.1`), freeze flags, fragment labels and Z-matrices. If there is a `--Link1--`, the structure of the first job is used.

### Output (contents of the ZIP)

| File | Contents |
|---|---|
| `<output name>_opt.xyz` | Optimized structure. Energy and settings in the comment line |
| `<output name>_opt.log` | All optimization steps in the format of Gaussian's opt output. GaussView shows the path animation and the energy plot |
| `<output name>_traj.xyz` | All optimization steps (multi-structure xyz) |
| `<output name>_log.txt` | Input settings, software versions, energy and fmax of each step |
| `<output name>_freq.txt` | List of frequencies (when vibrational analysis is on) |
| `<output name>_freq.log` | Frequencies and normal modes in the format of Gaussian's freq output (when vibrational analysis is on) |
| `<output name>_dft.csv` | DFT single point energy (when DFT single point is on) |

## ColabUMA Scan

[ColabUMA_Scan.ipynb](ColabUMA_Scan.ipynb) runs a relaxed scan: it moves a bond length, angle or dihedral in fixed steps and optimizes the remaining degrees of freedom. It corresponds to the `S` entry of Gaussian's `opt=modredundant`. Setup and usage are the same as for ColabUMA Opt.

| Item | Default | Description |
|---|---|---|
| `scan_atoms` | `1 2 3 4` | Atom numbers (starting at 1). 2 atoms for a bond length, 3 for an angle, 4 for a dihedral |
| `scan_start` | empty | Start value (Å or degrees). If empty, the value in the input structure (after the optimization, if the pre-scan optimization is on) |
| `scan_n_steps` | 36 | Number of steps. The number of points computed is this plus 1 |
| `scan_step_size` | 10 | Step size (Å or degrees). Negative to go the other way |
| `opt_before_scan` | On | Optimizes the input structure without constraints before the scan |
| `opt_after_scan` | On | Optimizes without constraints after the scan, starting from the first minimum found going back from the last point |
| Convergence | normal (0.02) | Same choices as ColabUMA Opt; only the default differs |

The optimization after the scan comes back from a scan that overshot and finds the minimum. Bumps smaller than 0.1 kcal/mol and unconverged points are ignored. A note is shown if the energy still decreases at the end of the scan, or if it goes all the way back to the first point.

If you don't know the atom numbers, run once: "5. Prepare the structures" shows a numbered 3D view.

| File | Contents |
|---|---|
| `<output name>_scan.csv` | Target value, measured value, energy, ΔE, fmax, steps and convergence of each point (unconverged points included) |
| `<output name>_scan.xyz` | Optimized structure of each converged point (multi-structure xyz) |
| `<output name>_scan.log` | Final structure of each converged point, in the format of Gaussian's scan output |
| `<output name>_preopt.xyz` / `.log` | Result and path of the optimization before the scan (when on) |
| `<output name>_postopt.xyz` / `.log` | Result and path of the optimization after the scan (when on) |
| `<output name>_log.txt` | Input settings, software versions, energy and fmax of each point and step |

## ColabUMA TS

[ColabUMA_TS.ipynb](ColabUMA_TS.ipynb) searches for the transition state (TS) of a reaction.

1. **DMF** (direct MaxFlux method, [PyDMF](https://github.com/shin1koda/dmf)) optimizes the reaction path between the reactant and the product, and the highest point on the path becomes the TS guess (the initial path is made with FB-ENM)
2. **Sella** optimizes it to a first-order saddle point. The initial Hessian is a numerical derivative of UMA
3. Vibrational analysis checks that there is exactly one imaginary frequency. The TS is displaced slightly along the imaginary mode in both directions and optimized, to check that it falls to structures with the same bonding as the reactant and the product (simple IRC check)
4. **DFT single points** ([GPU4PySCF](https://github.com/pyscf/gpu4pyscf), density fitting) of the reactant, TS and product at the UMA structures, and the UMA and DFT activation energies side by side

Give the reactant and the product as two files (xyz or gjf) with **the same atoms in the same order**. The `preopt.xyz` and `postopt.xyz` from a scan work as they are. If you already have a TS guess, choose "TS guess (Sella only)" to skip DMF.

| Item | Default | Description |
|---|---|---|
| `ts_input` | reactant and product (DMF → Sella) | "reactant and product (DMF → Sella)" or "TS guess (Sella only)" |
| `opt_endpoints` | On | Optimizes the reactant and the product before DMF |
| `dmf_nmove` | 5 | Number of points on the path where the energy is computed |
| `dmf_convergence` | middle | DMF convergence (loose / middle / tight) |
| `dmf_max_iter` | 300 | Maximum DMF iterations |
| `ts_exact_hessian` | On | Builds Sella's initial Hessian by numerical differentiation (6N force calls) |
| `check_connection` | On | Simple IRC check |
| `run_dft` | On | DFT single points |
| `dft_functional` / `dft_basis` | B3LYP-D3(BJ) / def2-SVP | The default favors speed. ωB97M-V/def2-TZVPD is the level of theory of UMA's training data (OMol25), but it is slow on a T4 |
| `dft_time_limit` | 60 | Time limit for DFT (minutes, total of reactant, TS and product). 0 means no limit |
| `gaussian_route` | `# opt=(ts,calcfc,noeigentest) freq b3lyp/def2svp empiricaldispersion=gd3bj` | Route line of the Gaussian gjf in the ZIP |

| File | Contents |
|---|---|
| `<output name>_ts.xyz` / `_ts.gjf` | TS structure, and a Gaussian input for a DFT TS optimization |
| `<output name>_ts_opt.log` | Sella optimization path (format of Gaussian's opt=ts) |
| `<output name>_ts_freq.log` / `.txt` | Frequencies and normal modes of the TS (GaussView animates the imaginary mode) |
| `<output name>_dmf_path.xyz` / `.log` / `.csv` | DMF reaction path and energies (GaussView shows the path animation) |
| `<output name>_dmf_init.xyz`, `_ts_guess.xyz` | FB-ENM initial path, highest point of DMF |
| `<output name>_reactant.xyz` / `_product.xyz` | End points (with `_opt.log` when they were optimized) |
| `<output name>_irc_minus.xyz` / `_irc_plus.xyz` | Structures reached by the simple IRC check |
| `<output name>_dft.csv` | DFT single point energies |

The TS found is a TS on the UMA potential energy surface. The DFT single points use the UMA structures, so finally run a DFT TS optimization and IRC (for example with the gjf). On the first run, the install takes a few minutes longer because IPOPT (cyipopt), needed by DMF, is built.

## ColabUMA Workflow

Colab gives each notebook its own session (its own machine). If you go ColabUMA_Opt → ColabUMA_Scan → ColabUMA_TS in separate notebooks, you redo the install and the model download each time.

[ColabUMA_Workflow.ipynb](ColabUMA_Workflow.ipynb) runs **optimization → scan → transition state → DFT single points** in a single session. Each stage passes its structures to the next in memory.

- The optimized structure becomes the reactant and the structure optimized after the scan becomes the product for DMF (set `ts_source` to "Sella from the scan maximum" to skip DMF and start from the highest point of the scan)
- To redo only part of the run, edit the form and run from that cell down (**Runtime → Run after**). The install and the model load are skipped
- All stages go into a single ZIP
- After the transition state stage, a plot overlays the DMF path, TS guess, Sella TS and product on the scan's energy curve. The x axis is the scanned coordinate measured in each structure
- DFT single points are off by default (turn on `run_dft`). The time limit `dft_time_limit` works as in the TS notebook
- The ZIP contains Gaussian gjf files for the TS (`_ts.gjf`, route line `gaussian_route`) and the IRC (`_irc.gjf`, route line `gaussian_irc_route`). Both use the UMA TS structure

## ColabDFT SPE

[ColabDFT_SPE.ipynb](ColabDFT_SPE.ipynb) runs only a DFT single point with GPU4PySCF, on the input structure as it is. It does not optimize the structure. It does not use UMA, so you don't need the Hugging Face token from [Before you start](#before-you-start-first-time-only), and fairchem is not installed.

| Item | Default | Description |
|---|---|---|
| Charge / spin multiplicity | 0 / 1 | Taken from the form, not from the gjf. Multiplicity 1 uses RKS, others UKS |
| Functional | B3LYP-D3(BJ) | ωB97M-V / B3LYP-D3(BJ) / PBE0-D3(BJ) / M06-2X / r2SCAN-D4 |
| Basis set | def2-SVP | def2-SVP / def2-TZVP / def2-TZVPD |

If an xyz holds several structures, only the first is computed. Supported elements are H–Rn (atomic numbers 1–86).

| File | Contents |
|---|---|
| `<output name>_spe.xyz` | The computed structure. Energy and settings in the comment line |
| `<output name>_dft.csv` | DFT single point energy |
| `<output name>_log.txt` | Input settings, software versions, results |

## ColabDFT ESP

[ColabDFT_ESP.ipynb](ColabDFT_ESP.ipynb) runs a DFT single point on the input structure as it is, and shows the electrostatic potential (ESP) colored on an electron density isosurface in 3D. Red is negative, blue is positive. It does not optimize the structure, and needs neither UMA nor a token. The charge, multiplicity, functional and basis set items are the same as in [ColabDFT SPE](#colabdft-spe).

| Item | Default | Description |
|---|---|---|
| `input_mode` | file | file: choose an xyz or gjf. draw: draw the structure in an editor (JSME); hydrogens are added and RDKit makes the 3D structure (10 ETKDG conformers optimized with MMFF94, lowest one kept). The structure is not optimized further |
| `isosurface` | 0.002 | Density of the isosurface that is colored (e/Bohr³) |
| `grid_quality` | standard | Grid spacing. coarse 0.5 / standard 0.35 / fine 0.25 Bohr. Made coarser automatically for large molecules |
| `esp_color_range` | 0 | Color range ± (kcal/mol). 0 means automatic. Use the same value when comparing molecules |

| File | Contents |
|---|---|
| `<output name>_esp.cube` | ESP on the grid (a.u.) |
| `<output name>_density.cube` | Electron density on the grid (e/Bohr³). In GaussView, color an isosurface of this with `_esp.cube` |
| `<output name>_spe.xyz` | The computed structure. Energy and settings in the comment line |
| `<output name>_dft.csv` | DFT single point energy |
| `<output name>_log.txt` | Input settings, software versions, results, minimum and maximum ESP on the surface |

## ColabDFT MO

[ColabDFT_MO.ipynb](ColabDFT_MO.ipynb) runs a DFT single point on the input structure as it is, and shows molecular orbitals (Kohn–Sham orbitals) such as HOMO and LUMO as positive and negative isosurfaces in 3D. Blue is positive, red is negative (the overall sign of an orbital has no meaning, so the colors can swap). An energy level diagram (occupations shown as ↑↓) sits next to the 3D view; click a level to show that orbital. Orbitals not given in the form are computed on click by the Colab runtime (only while it is connected). It also shows the HOMO and LUMO energies, the gap, and a table of the orbital energies near HOMO and LUMO. Only the orbitals given in the form are written as cube files to the ZIP. For open shells, α and β orbitals are shown separately. It does not optimize the structure, and needs neither UMA nor a token. The charge, multiplicity, functional and basis set items are the same as in [ColabDFT SPE](#colabdft-spe).

| Item | Default | Description |
|---|---|---|
| `input_mode` | file | file: choose an xyz or gjf. draw: draw the structure in an editor (JSME); hydrogens are added and RDKit makes the 3D structure (10 ETKDG conformers optimized with MMFF94, lowest one kept). The structure is not optimized further |
| `orbitals` | HOMO, LUMO | Orbitals to show, separated by commas. `HOMO-1`, `LUMO+2` and orbital numbers (from 1, lowest energy first) also work. Up to 8 at a time |
| `isovalue` | 0.02 | Isosurface value ± (Bohr^-3/2). Same as the GaussView default |
| `grid_quality` | standard | Grid spacing. coarse 0.5 / standard 0.35 / fine 0.25 Bohr. Made coarser automatically for large molecules |

| File | Contents |
|---|---|
| `<output name>_mo<number>_<orbital>.cube` | Orbital values on the grid (Bohr^-3/2). For example `_mo5_homo.cube`, `_mo4_homom1.cube` (HOMO-1), `_mo7_lumop1.cube` (LUMO+1). Open shells get `_alpha` / `_beta` |
| `<output name>_mo_energies.csv` | Energies (Hartree and eV) and occupations of all orbitals |
| `<output name>_spe.xyz` | The computed structure. Energy and settings in the comment line |
| `<output name>_dft.csv` | DFT single point energy |
| `<output name>_log.txt` | Input settings, software versions, results, HOMO and LUMO energies and the gap |

## DFT memory and time

In the UMA notebooks (ColabUMA Opt, TS and Workflow), if the DFT single points fail, **only DFT is skipped and the notebook goes on to save the UMA results**. The reason is shown as a note and written to `log.txt`. Energies of the structures that finished before the failure are kept in `dft.csv`.

DFT is skipped when:

- **Memory is expected to run out.** Before starting, the notebook estimates the size of the three-center integrals for density fitting. gpu4pyscf keeps them in GPU memory, or in CPU memory if they do not fit. If CPU memory also runs out, Colab does not raise an error: the runtime crashes and the UMA results are lost too. When that looks likely, DFT is skipped without starting
- **Memory runs out during the calculation** (for example a GPU out-of-memory error)
- **The time limit is exceeded.** When the DFT time limit (default 60 min, total over all structures) has passed, the calculation stops, even in the middle of an SCF. If the next structure would take as long as the previous one and cannot finish in time, it is not started
- **The Colab runtime is about to end.** A free Colab runtime lasts at most about 12 hours. DFT that would not finish 15 minutes before 12 hours from the start of the runtime is stopped (even with the limit set to 0)
- Any other error occurs in the DFT calculation

**Memory warnings:** when memory looks tight, a note is shown in the structure preparation cell (before the UMA calculations start) and in the DFT cell. The size of the three-center integrals is an upper bound; it is often smaller in practice because small integrals are screened out. As a guide, on a T4 (15 GB GPU, about 12 GB CPU memory), for a molecule of about 80 atoms, def2-SVP fits in GPU memory, while def2-TZVP and larger are expected to exceed CPU memory too, so DFT is skipped.

ColabDFT SPE, ESP and MO exist to run DFT, so they do not skip it. They only show a note when memory looks tight.

About Colab timeouts:

- A free runtime lasts at most about 12 hours (longer with Colab Pro)
- If you don't interact with the browser for a long time, the runtime can disconnect even while a calculation runs. Keep the tab open during long calculations
- When the runtime ends, all results on the Colab machine are lost. Download the ZIP after each run

## Code layout

Notebook cells only take the form values and call functions. The calculations live in the [`colabuma/`](colabuma) package of this repository. In "2. File selection and checks", each notebook fetches this repository into Colab (`git clone`) and imports it. Adding a notebook does not duplicate the shared code, and a fix goes in one place.

| Module | Contents |
|---|---|
| `env.py` | GPU and token checks, install, Hugging Face login, model load (once per session) |
| `structure.py`, `draw.html` | Upload, drawing input (JSME → 3D with RDKit), reading xyz and gjf, charge and multiplicity checks, 3D view |
| `opt.py` | Optimization and progress bars, vibrational analysis (numerical Hessian), results of the Opt notebook |
| `scan.py` | Relaxed scan |
| `ts.py` | DMF, Sella, TS checks, simple IRC check |
| `dft.py` | DFT single points (gpu4pyscf, or CPU PySCF if the GPU is unavailable), memory estimate, time limit |
| `esp.py` | Electron density and ESP on a grid, cube output, isosurface view |
| `mo.py`, `mo_explorer.html` | Molecular orbitals on a grid, orbital energies, clickable level diagram with positive and negative isosurfaces |
| `gaussian.py` | Writing Gaussian logs (opt / scan / freq / path) and gjf files |
| `output.py` | log.txt and the ZIP |

To try code from another branch, set `COLABUMA_REF` in cell "2." to that branch name (restart the session if the code was already fetched). If you clone the repository locally and open a notebook from it, the `colabuma` in that folder is used.

The notebooks normally use the latest code on main. To reproduce the version used in a paper, set `COLABUMA_REF` to a release tag (for example `v0.3.0`). The version and commit used are written under "ソフトウェア" (software) in `log.txt`.

## Notes

- An optimization can converge to a saddle point. Check important structures with a vibrational analysis.
- UMA is a machine learning potential and does not replace DFT. Confirm final energy comparisons with DFT.
- For small molecules (up to a few tens of atoms), using DFT directly is often more reliable.
- Supported elements are H–Bi (atomic numbers 1–83).
- `_opt.log` imitates Gaussian's format for viewing; it is not a Gaussian result. Energies (Hartree) and forces are UMA values. The threshold column holds the form's fmax converted to Hartree/Bohr, with the same ratios as Gaussian's defaults.
- The vibrational analysis builds the Hessian from central differences of UMA forces (0.01 Å), projects out translations and rotations, and diagonalizes. Imaginary frequencies with an absolute value under 20 cm⁻¹ are treated as numerical noise and not counted (`freq.txt` also lists the values before projection).

## Acknowledgments

ColabChem draws heavily on [ColabReaction](https://github.com/BILAB/ColabReaction) (Bioinformation Engineering Laboratory, The University of Tokyo; [ColabReaction.net](https://colabreaction.net/)). The idea of searching for transition states on Google Colab with a machine learning potential (UMA) and Direct MaxFlux (DMF), and much of the notebook design, follow ColabReaction.

- M. Karasawa, C. S. Leow, H. Yajima, S. Arai, H. Nishizaki, T. Terada, H. Sato, "ColabReaction: Accelerating Transition State Searches with Machine Learning Potentials on Google Colaboratory", *J. Chem. Inf. Model.* **65**, 11908–11914 (2025). https://doi.org/10.1021/acs.jcim.5c02398

## Citation

If you use ColabChem in a publication, please cite ColabChem together with the methods you used. The citation information for ColabChem is in [CITATION.cff](CITATION.cff); you can copy it in APA or BibTeX format from "Cite this repository" on the right side of the GitHub repository page.

- R. Chiba, ColabChem, Zenodo, https://doi.org/10.5281/zenodo.24035698 (this DOI covers all versions; each version has its own DOI on the Zenodo page, v0.3.0 is 10.5281/zenodo.24035699)
- A. H. Larsen *et al.*, *J. Phys.: Condens. Matter* **29**, 273002 (2017). (ASE, used by all notebooks)

For the notebooks that use UMA (ColabUMA), also cite (not needed for ColabDFT, which does not use UMA):

- B. M. Wood *et al.*, "UMA: A Family of Universal Models for Atoms", arXiv:2506.23971 (2025).
- D. S. Levine *et al.*, "The Open Molecules 2025 (OMol25) Dataset, Evaluations, and Models", arXiv:2505.08762 (2025).
- fairchem: https://github.com/facebookresearch/fairchem

If you used the transition state search of ColabUMA TS, also cite:

- S.-i. Koda and S. Saito, *J. Open Source Softw.* **11**, 10379 (2026). (PyDMF)
- S.-i. Koda and S. Saito, *J. Chem. Theory Comput.* **20**, 2798–2811 (2024). (Direct MaxFlux)
- S.-i. Koda and S. Saito, *J. Chem. Theory Comput.* **20**, 7176–7187 (2024); **21**, 3513–3522 (2025). (FB-ENM, CFB-ENM)
- E. D. Hermes, K. Sargsyan, H. N. Najm, J. Zádor, *J. Chem. Theory Comput.* **18**, 6974–6988 (2022). (Sella)

If you used DFT single points (ColabUMA Opt, TS, Workflow, ColabDFT), also cite:

- PySCF (Q. Sun *et al.*, *J. Chem. Phys.* **153**, 024109 (2020))
- GPU4PySCF (X. Wu *et al.*, arXiv:2404.09452 (2024))

ColabChem was developed with AI coding assistance (Claude Code).

## License

- Notebooks and shared code (`colabuma/`): [MIT License](LICENSE)
- UMA model weights: under the [FAIR Chemistry License v1](https://huggingface.co/facebook/UMA). The weights are not included in this repository; each user downloads them from Hugging Face after accepting the license

### Software installed at runtime

This repository does not include the software below. When a notebook runs, it is installed into the Colab session from PyPI or Ubuntu packages (apt), and used under its own license.

| Software | Notebooks | License |
|---|---|---|
| [fairchem-core](https://github.com/facebookresearch/fairchem) | All but ColabDFT | MIT |
| [ASE](https://gitlab.com/ase/ase) | All | LGPL-2.1-or-later |
| [py3Dmol](https://github.com/3dmol/3Dmol.js) | All | MIT |
| [PyDMF](https://github.com/shin1koda/dmf) (`pydmf`) | TS, Workflow | MIT |
| [Sella](https://github.com/zadorlab/sella) | TS, Workflow | LGPL-3.0 |
| [cyipopt](https://github.com/mechmotum/cyipopt) | TS, Workflow | EPL-2.0 |
| [Ipopt](https://github.com/coin-or/Ipopt) (apt `coinor-libipopt-dev`, 3.11 series) | TS, Workflow | CPL-1.0 (EPL-2.0 from 3.13) |
| MUMPS (linear solver used by Ipopt, installed with it by apt) | TS, Workflow | CeCILL-C |
| [GPU4PySCF](https://github.com/pyscf/gpu4pyscf) (`gpu4pyscf-cuda12x`, `gpu4pyscf-libxc-cuda12x`) | Opt, TS, Workflow (DFT), ColabDFT | Apache-2.0 |
| [CuPy](https://cupy.dev/) (`cupy-cuda12x`) | Opt, TS, Workflow (DFT), ColabDFT | MIT |
| [cuTENSOR](https://developer.nvidia.com/cutensor) (`cutensor-cu12`) | Opt, TS, Workflow (DFT), ColabDFT | NVIDIA proprietary license |
| [PySCF](https://github.com/pyscf/pyscf) | Opt, TS, Workflow (DFT), ColabDFT | Apache-2.0 |
| [pyscf-dispersion](https://github.com/pyscf/dispersion) | Opt, TS, Workflow (DFT), ColabDFT | Apache-2.0 |
| [RDKit](https://github.com/rdkit/rdkit) | ColabDFT ESP and MO (only with the draw input) | BSD-3-Clause |
| [JSME](https://jsme-editor.github.io/) (structure editor; not installed, the browser loads it from jsDelivr when shown) | ColabDFT ESP and MO (only with the draw input) | BSD-3-Clause |

These licenses are as stated by each package's distribution (PyPI and Ubuntu packages) as of November 2026. Libraries they depend on (PyTorch, NumPy, SciPy, etc.) are installed as well. Check each distribution for the exact terms.
