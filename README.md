hypergeometric
==============

Scripts for calculating Wallenius noncentral hypergeometric distributions and their application to statistical distribution of modifier in aluminoborosilicate glasses. In analogy with the classic problem of drawing marbles from an urn, each alkali atom is said to "draw" a particular site, one at a time, without replacement, until all available alkali are exhausted. Based on the theory presented in "Statistics of modifier distributions in mixed network glasses" (2013, Mauro) <http://dx.doi.org/10.1063/1.4773356>.

hypergeometric.py contains a recursive implementation of the multivariate noncentral Wallenius hypergeometric distribution, and three different approaches for estimating alkali distribution in aluminoborosilicate glasses. In each approach, for a given composition or set of compositions and given values of relative enthalpy associated with drawing each type of site, expectation values are calculated estimating the relative fraction of modifier associated with each type of site.

Fitting to data
---------------

The other three included .py files apply these three expectation value calculations to the composition series given in "Structure of boroaluminosilicate glasses: Impact of [Al2O3]/[SiO2] ratio on the structural role of sodium" (2012, Zheng et al.) <http://dx.doi.org/10.1103/PhysRevB.86.054203>. Calculated N4 are fit against the measured N4 provided in Table 1. The relative enthalpies of each type of draw are treated as fitting parameters. The resulting best fit is then plotted against the source data to yield a result comparable to Fig. 11 in the paper.

Details and example plots for each of these scripts are summarized below.


Univariate N4, Q3
----------------

This simplest approach calculates expectation values for fraction of tetrahedral boron (N4) and fraction of Si tetrahedra with a nonbridging oxygen (Q3). Alumina is assumed to preferentially consume available modifier until it is fully converted to tetrahedral
units, so Al is not considered in the statistics. Since modifier is only partitioned between two possible outcomes, the univariate Wallenius distribution is used.

![Univariate N4, Q3 best fit](/plots/Univariate_N4_Q3.png)


Bivariate N4, L4, Q3
--------------------

This approach calculates expectation values for fraction of tetrahedral boron (N4), fraction of tetrahedral aluminum (L4), and fraction of Si tetrahedra with a nonbridging oxygen (Q3). In this case no assumptions about the behavior of Al are used, tetrahedral conversion of alumina units is treated as as a third type of site that can be drawn by the modifier. Since modifier is partitioned between three possible outcomes, the bivariate Wallenius distribution is used.

![Bivariate N4, L4, Q3 best fit](/plots/Bivariate_N4_L4_Q3.png)


Bivariate N4, N2, Q3
--------------------

This approach calculates expectation values for fraction of tetrahedral boron (N4), fraction of trigonal boron with a nonbridging oxygen (N2), and fraction of Si tetrahedra with a nonbridging oxygen (Q3). Alumina is again assumed to preferentially consume available modifier until it is fully converted to tetrahedral units, and is not considered in the statistics. Since modifier is partitioned between three possible outcomes, the bivariate Wallenius distribution is used.

![Bivariate N4, N2, Q3 best fit](/plots/Bivariate_N4_N2_Q3.png)

It should be noted that although this appears to produce the best fit, it also relies on an additional fitting parameter and may not be physically meaningful. Indeed, this fit suggests that at low Al2O3 content, *all* trigonal boron is converted to either tetrahedral units (N4) or partially depolymerized (N2) units with a nonbridging oxygen, which is not the expected behavior. This anomaly could indicate that other more important mechanisms exist which have not been included in the model (such as the 'avoidance rules' of the Dell-Bray model), and the fitting parameter intended to model NBO-on-B is capturing these effects.

Dependencies
------------

Numpy, Scipy, Matplotlib


Further reading
---------------

A discussion of the details of each function used in this implementation can be found in the source code. Resources I used to understand the problem and develop a generalized solution are included below. My implementation can be considered a multivariate generalization of the univariate recursive solution described in the first paper by Fog.

  An overview of the theory and analysis of its application to a model glass system with two network formers is presented in "Statistics of modifier distributions in mixed network glasses" (2013, Mauro) <http://dx.doi.org/10.1063/1.4773356>

  An overview of the Wallenius noncentral hypergeometric distribution:
  http://en.wikipedia.org/wiki/Wallenius'_noncentral_hypergeometric_distribution

  A. Fog - Calculation Methods for Wallenius' Noncentral Hypergeometric Distribution
  <http://www.agner.org/random/theory/nchyp1.pdf>

  A. Fog - Sampling Methods for Wallenius' and Fisher's Noncentral Hypergeometric Distributions <http://www.agner.org/random/theory/nchyp2.pdf>

  A. Fog - Biased Urn Theory <http://cran.r-project.org/web/packages/BiasedUrn/vignettes/UrnTheory.pdf>


  License
  -------

  MIT
