# PIMEX

## Physics-Informed Modular Excavation Framework

PIMEX develops modular physics-informed models for underground excavation, with a focus on tunnel ground deformation, lining response, and coupled ground–structure behaviour. The longer-term aim is to support parameter identification and digital twins of underground infrastructure.

This repository is the central project hub. Individual models will be maintained in separate repositories and linked here as they are added.

## Model repository index

The names below are proposed names; the model repositories have not yet been linked.

| Module | Scope | Proposed repository | Link |
| --- | --- | --- | --- |
| Ground model | Ground displacement, stress, and strain; linear-elastic baseline followed by Mohr–Coulomb extensions | `PIMEX-Ground` | To be added |
| Tunnel lining model | Circular lining deformation and internal forces using curved-beam formulations, including HRM/VHRM-inspired approaches | `PIMEX-Lining` | To be added |
| Coupled ground–lining model | Modular coupling through interface displacement compatibility and traction equilibrium | `PIMEX-Coupled` | To be added |

Additional model repositories can be added to this index as the project develops.

## Research roadmap

1. Verify the ground and lining models independently.
2. Establish a 2D coupled linear-elastic ground and elastic circular lining baseline with a bonded interface.
3. Validate forward predictions independently against analytical solutions, HRM, and numerical benchmarks such as PLAXIS.
4. Develop inverse models using sparse observations to identify parameters, starting with soil Young’s modulus and assessing identifiability.
5. Extend the framework to interface contact, separation and slip, followed by incremental Mohr–Coulomb ground plasticity.

## Modelling principles

- Train the forward PINN using physics: equilibrium, constitutive relations, boundary conditions, lining periodicity, and interface conditions.
- Use benchmark data for independent validation of the forward model.
- Keep inverse identification separate, with observation and holdout validation sets.
- When coupling an explicit ground continuum to the lining, avoid duplicating ground resistance through additional HRM soil springs.
- Document assumptions, units, loading stages, reproducible configurations, and validation results within each model repository.

## Adding a model repository

Create the model repository under this GitHub account, add its purpose and implementation details to its own README, then replace the corresponding “To be added” entry above with a link. Each repository should document its inputs, outputs, governing equations, training losses, setup instructions, and benchmark results.

## Project context

Developed by **Maziyar Bahri** at **Durham University**, working with **Jelena Ninić** on the PIMEX research project.

**Status:** Research in development. This hub currently provides the project overview and repository roadmap; model code will be added through the linked repositories.
