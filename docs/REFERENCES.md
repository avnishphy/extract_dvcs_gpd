# References

The machine-readable figure-family inventory and convention gates are in
[`configs/literature/benchmarks_v1.json`](../configs/literature/benchmarks_v1.json)
and summarized in [Literature benchmarks](LITERATURE_BENCHMARKS.md). It includes
Xu et al., arXiv:2605.06994; Moffat et al., arXiv:2303.12006; and Bertone et
al., arXiv:2107.11312. No paper figures or invented digitized curves are stored.

## GPD and DVCS foundations

1. M. Diehl, “Generalized Parton Distributions,” *Physics Reports* **388** (2003) 41–277. arXiv:hep-ph/0307382. DOI: 10.1016/j.physrep.2003.08.002.
2. A. V. Radyushkin, “Generalized Parton Distributions,” arXiv:hep-ph/0101225.
2a. A. V. Radyushkin, “Double Distributions and Pseudo-Distributions,” arXiv:2311.06007v2. Equations (2.1), (3.4), (3.5), and (5.4) are the production DD, forward-limit, polynomiality, and normalized-profile basis. https://arxiv.org/html/2311.06007v2
3. D. Müller and A. Schäfer, “Complex conformal spin partial wave expansion of generalized parton distributions and distribution amplitudes,” *Nucl. Phys. B* **739** (2006) 1–59. arXiv:hep-ph/0509204.
3a. Y. Guo, X. Ji, and K. Shiells, “Generalized parton distributions through universal moment parameterization: zero skewness case,” *JHEP* **09** (2022) 215. arXiv:2207.05768. DOI: 10.1007/JHEP09(2022)215.
3b. H.-C. Zhang and X. Ji, “On convergence properties of GPD expansion through Mellin/conformal moments and orthogonal polynomials,” *Nucl. Phys. B* **1010** (2025) 116762. arXiv:2408.04133. DOI: 10.1016/j.nuclphysb.2024.116762.
4. P. V. Pobylitsa, “Virtual Compton scattering in the generalized Bjorken region and positivity bounds on generalized parton distributions,” *Phys. Rev. D* **66** (2002) 094002. arXiv:hep-ph/0211160.

## PARTONS and evolution software

5. B. Berthou et al., “PARTONS: PARtonic Tomography Of Nucleon Software,” *Eur. Phys. J. C* **78** (2018) 478. DOI: 10.1140/epjc/s10052-018-5948-0. Official documentation: https://partons.cea.fr/partons/doc/html/
6. V. Bertone, “APFEL++: A new PDF evolution library in C++,” arXiv:1708.00911. Repository: https://github.com/vbertone/apfelxx
7. V. Bertone et al., “Revisiting evolution equations for generalised parton distributions,” *Eur. Phys. J. C* **82** (2022) 888. arXiv:2206.01412. DOI: 10.1140/epjc/s10052-022-10793-0.
8. V. Bertone et al., “One-loop evolution of twist-2 generalized parton distributions,” *Phys. Rev. D* **109** (2024) 034023. arXiv:2311.13941. DOI: 10.1103/PhysRevD.109.034023.
9. A. Buckley et al., “LHAPDF6: parton density access in the LHC precision era,” *Eur. Phys. J. C* **75** (2015) 132. arXiv:1412.7420. DOI: 10.1140/epjc/s10052-015-3318-8.

## Neural CFF/GPD inference and deconvolution

10. K. Kumerički, D. Müller, and A. Schäfer, “Neural network generated parametrizations of deeply virtual Compton form factors,” *JHEP* **07** (2011) 073. arXiv:1106.2808.
11. H. Moutarde, P. Sznajder, and J. Wagner, “Unbiased determination of DVCS Compton Form Factors,” *Eur. Phys. J. C* **79** (2019) 614. arXiv:1905.02089. DOI: 10.1140/epjc/s10052-019-7117-5.
12. H. Dutrieux, O. Grocholski, H. Moutarde, and P. Sznajder, “Artificial neural network modelling of generalised parton distributions,” *Eur. Phys. J. C* **82** (2022) 252. arXiv:2112.10528. DOI: 10.1140/epjc/s10052-022-10211-5; see also the published erratum.
13. V. Bertone, H. Dutrieux, C. Mezrag, H. Moutarde, and P. Sznajder, “The deconvolution problem of deeply virtual Compton scattering,” *Phys. Rev. D* **103** (2021) 114019. arXiv:2104.03836. DOI: 10.1103/PhysRevD.103.114019.
14. Y. Guo et al., “Generalized parton distributions through universal moment parameterization: non-zero skewness case,” arXiv:2302.07279.
15. Y. Guo et al., “GUMP1.0—First global extraction of generalized parton distributions,” arXiv:2509.08037.
16. J. Xu et al., “Neural Network Representation of Generalized Parton Distributions,” arXiv:2605.06994.

## Simulation-based inference

17. K. Cranmer, J. Brehmer, and G. Louppe, “The frontier of simulation-based inference,” *PNAS* **117** (2020) 30055–30062. arXiv:1911.01429. DOI: 10.1073/pnas.1912789117.
18. D. S. Greenberg, M. Nonnenmacher, and J. H. Macke, “Automatic Posterior Transformation for Likelihood-Free Inference,” *ICML/PMLR* **97** (2019) 2404–2414. arXiv:1905.07488.
19. C. Durkan, A. Bekasov, I. Murray, and G. Papamakarios, “Neural Spline Flows,” *NeurIPS* (2019). arXiv:1906.04032.
20. `sbi` official documentation and repository:
    https://sbi-dev.github.io/sbi/ and https://github.com/sbi-dev/sbi.
    The distribution pins release 0.26.1.
20c. PyTorch, official CUDA availability and CUDA semantics documentation:
    https://docs.pytorch.org/docs/stable/generated/torch.cuda.is_available.html
    and https://docs.pytorch.org/docs/stable/notes/cuda.html.
    The official 2.12.1 wheel matrix used to select CPU/CUDA 12.6 distribution wheels is:
    https://pytorch.org/get-started/previous-versions/.
20d. Optuna, official installation and `create_study` documentation:
    https://optuna.readthedocs.io/en/stable/installation.html and
    https://optuna.readthedocs.io/en/stable/reference/generated/optuna.create_study.html.
    The distribution pins Optuna 4.5.0.

## Conventional posterior diagnostics

20a. N. Metropolis et al., “Equation of State Calculations by Fast Computing Machines,” *J. Chem. Phys.* **21** (1953) 1087–1092. DOI: 10.1063/1.1699114.
20b. A. Gelman and D. B. Rubin, “Inference from Iterative Simulation Using Multiple Sequences,” *Statistical Science* **7** (1992) 457–472. DOI: 10.1214/ss/1177011136.

## Reference policy

Prefer primary papers and official project documentation. A citation describes theory or software context; the installed source commit, capability response, local tests, and dependency lock remain authoritative for the exact behavior shipped here. Users publishing results should cite this distribution, PARTONS, APFEL++, LHAPDF and the selected PDF set when used, sbi/PyTorch, the relevant GPD/DVCS theory, and every experimental dataset shown.
