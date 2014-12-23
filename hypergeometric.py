from __future__ import division
import numpy as np
import csv
import traceback

kB = 8.6173324e-5 # eV/K



#=============================================================================#
#================================= UTILITY ===================================#
#=============================================================================#

        
        
        

class Memoize(object):
    """
    Enable functions to cache results to prevent redundant calculations
    during recursion (massive performance boost).

    Usage:
    >>> def f_recursive(x):
    >>>    ...
    >>>
    >>> f_recursive = Memoize(f_recursive)
    """
    def __init__(self, fn):
        self.fn = fn
        self.memo = {}

    def __call__(self, *args):
        # cast to hashable form
        args = tuple(args)
        if not self.memo.has_key(args):
            self.memo[args] = self.fn(*args)
        return self.memo[args]

    def reset(self):
        self.memo = {}




def get_data(csv_filename):
    with open(csv_filename, 'rb') as csvfile:
        return [row for row in csv.reader(csvfile, delimiter=',')]




def get_col(raw_data, col):
    data = []
    for row in raw_data[1:]:
        if row[col] != '':
            data.append(float(row[col]))
    return np.array(data)



#=============================================================================#
#============================== PROBABILITIES ================================#
#=============================================================================#



def p_draw(index, totals_drawn, initial_populations, weights, same_type=None):
    """
    Probability that the next draw selects the [int: index] type, given 
    the [array-like: totals_drawn] so far, [array-like: initial_populations], 
    and [array-like: weights] of each type.

    If two types of draws can occur for the same unit (e.g. trigonal 
    boron could be converted to tetrahedral (N4) or could be assigned 
    NBO (N2)), same_type should be an array-like containing the indices of 
    these two types within the other provided arrays so that the same 
    population will be considered for both.
    """
    k, n, w = [np.array(arg) for arg in
              [totals_drawn, initial_populations, weights]]
    if same_type == None:
        p = (n-k)*w / np.dot(n-k, w)
        return p[index]
    else:
        terms = []
        sites = zip(k, n, w)
        for i, (n, g, w) in enumerate(sites):
            if i in same_type:
                ni0 = sites[same_type[0]][0]
                ni1 = sites[same_type[1]][0]
                terms.append((g - ni0 - ni1)*w)
            else:
                terms.append((g - n)*w)
        return terms[index]/sum(terms)





def p_totals_drawn(totals_drawn, initial_populations, weights, same_type=None):
    """
    Probability that exactly [array-like: totals_drawn] units of each type 
    will be drawn after total_draws, given the [array-like: initial_populations]
    and [array-like: weights] of each type of site.

    If two types of draws can occur for the same unit (e.g. trigonal 
    boron could be converted to tetrahedral (N4) or could be assigned 
    NBO (N2)), same_type should be an array-like containing the indices of 
    these two types within the other provided arrays so that the same 
    population will be considered for both.

    Details
    =======

    This recursive function calculates the probability of one specific
    outcome, which is used in calculating the full probability distribution
    (individual probabilities of all possible outcomes) and from that,
    the expectation value (most probable outcome).

    This function should be Memoized to dramatically speed up calculation
    of the probability distribution. 

    Applicable for distribution of one modifier between multiple atomic
    sites in glass (e.g. Na distributed among Al, B, and Si sites).

    Algorithm rationale:
    --------------------
    For k types of draws, there are up to k routes to arrive at a 
    given set of total draws (n0 ... nk). The sum of probabilities 
    of each route gives the total probability of reaching the 
    situation (n0 ... nk).

    For example, for three types of draws
    (i, j, k) there are up to three routes:

      p(r1) = p(nj and nk reached last draw) * p(ni reached this draw)
      p(r2) = p(ni and nk reached last draw) * p(nj reached this draw)
      p(r3) = p(ni and nj reached last draw) * p(nk reached this draw)

    The 'last draw' situation for each route is itself arrived at by 
    up to k possible routes, so the calculation becomes recursive.

    Fewer than k routes occur in the edge cases where at least one 
    (ni ... nk) is zero (e.g. there is only one route to (1, 0, 0); 
    routes 2 and 3 above are not applicable). In the present 
    implementation all k routes are always considered, but the 
    invalid routes are recognized as 'impossible situations' and 
    return a probability of 0.
    """

    # rearrange arguments into (n, g, w) of each type
    types = zip(totals_drawn, initial_populations, weights)

    # catch impossible situations
    for n, g, w in types:
        if n < 0 or n > g:
            return 0
        if g == 0 and n != 0:
            return 0

    # initial condition, bottom of recursion
    if sum(totals_drawn) == 0:
        return 1

    # on each draw
    else:
        k = len(initial_populations)

        p = 0
        for route in range(k):
            prev_totals = list(totals_drawn)
            prev_totals[route] -= 1
            prev_totals = tuple(prev_totals)

            # probability last draw was consistent with this route
            p_last = p_totals_drawn(prev_totals, initial_populations, weights, same_type)

            # probability this draw is consistent with this route
            p_this = p_draw(route, prev_totals, initial_populations, weights, same_type)

            p += p_last * p_this

        return p

p_totals_drawn = Memoize(p_totals_drawn)





def P_distribution(total_draws, initial_populations, weights, same_type=None):
    """
    Probability distribution of p_totals_drawn() for all possible 
    combinations of totals_drawn that sum to [int: total_draws], 
    given the [array-like: initial_populations] and [array-like: weights] 
    of each type of site.

    If two types of draws can occur for the same unit (e.g. trigonal 
    boron could be converted to tetrahedral (N4) or could be assigned 
    NBO (N2)), same_type should be an array-like containing the indices of 
    these two types within the other provided arrays so that the same 
    population will be considered for both.

    Details
    =======

    This function calculates the probabilities of all possible outcomes,
    which are used to calculate the expectation value. The result is a 
    matrix of probabilities in k-1 dimensions, where k is the number of 
    types of draws.

    Note that k-1 dimensions are sufficient to represent all nontrivial 
    values: Once total_draws and the numbers of draws for each species 
    (n0 ... nk-1) are specified, nk is uniquely determined 
    (nk = total_draws - sum(n0 ... nk-1)).

    The recursive function p_totals_drawn should be Memoized to 
    dramatically reduce calculation time.

    Applicable for distribution of one modifier between multiple atomic 
    sites in glass (e.g. Na distributed among Al, B, and Si sites).
    """

    # dump old cache to prevent running out of memory during curve fitting
    p_totals_drawn.reset()

    P = np.zeros([total_draws + 1]*(len(initial_populations)-1))

    for totals_drawn in np.ndindex(P.shape):
        totals_drawn = [n for n in totals_drawn]
        totals_drawn.append(total_draws - sum(totals_drawn))
        totals_drawn = tuple(totals_drawn)
        P[totals_drawn[:-1][::-1]] = p_totals_drawn(
                totals_drawn, initial_populations, weights, same_type)

    # consistency check
    if np.abs(np.sum(P) - 1.0) > 0.01:
        print 'Warning: total probability of all possible outcomes ' + \
                'deviates from 1: {} !!'.format(1.0 - np.sum(P))

    return P





def expectation(total_draws, P_distribution):
    """
    Expectation values of total draws of each draw type associated 
    with the given probability distribution [array: P_distribution]. 
    The result is a 1D array with length equal to the number of types
    of draws. For example, in a system with three draw types i, j, k, 
    the expectation E will take the form [Ei, Ej, Ek] with 
    sum([Ei, Ej, Ek]) == [int: total_draws].

    As expectation is calculated by weighted averages of individual 
    probabilities, the results are not integers even though actual 
    draw outcomes can only be integers. If used to predict the most 
    likely actual outcome of a specific number of draws and specific 
    population sizes, the values should be rounded.

    In the case of modifier distribution in glass it is presumed that 
    the system can be scaled to arbitrary size and expressed in terms
    of a normalized composition (e.g. as mol %s of individual oxides). 
    Likewise the modifier distribution of interest is only the relative 
    fraction of modifier associated with each type of site, and in this 
    case integer values are not expected. The explicit calculation of a 
    specific number of draws is thus abstracted to the general case by 
    normalizing the result, i.e. dividing the expected values of each 
    type of draw by the number of total draws to obtain the relative 
    population fractions of each type.

    In practice systems involving up to about 150 draws can be
    calculated before exceeding the (default) maximum recursion depth. 
    In application to modifier distribution in glasses, this means 
    systems of only a few hundred or thousand total atoms can be 
    considered, however the relative modifier distribution remains 
    approximately constant when the size of the system is scaled.

    Example:
    >>> total_draws = 3
    >>> initial_populations = (6,3,2)
    >>> weights = (1,10,100)
    >>>
    >>> P = P_distribution(total_draws, initial_populations, weights)
    >>> Eo = expectation(total_draws, P)/total_draws
    >>> print Eo
    [ 0.06249871  0.3032176   0.63428369]

    Scaled by 10:
    >>> total_draws = 30
    >>> initial_populations = (60,30,20)
    >>> ...
    >>> print Eo
    [ 0.06677376  0.29087475  0.6423515 ]

    Scaled by 50:
    >>> total_draws = 150
    >>> initial_populations = (300,150,100)
    >>> ...
    >>> print Eo
    [ 0.06691752  0.28903773  0.64404475]

    """

    grids = np.mgrid[[slice(0, axis) for axis in P_distribution.shape]]
    E = [np.sum(grid*P_distribution) for grid in grids[::-1]]
    return np.array(E + [total_draws - sum(E)])





def chi_sq(data, best_fit):
    return np.sum((data - best_fit)**2)





#=============================================================================#
#================================ GLASS MODELS ===============================#
#=============================================================================#

def N4_Q3_expectation(TG, M2O, B2O3, SiO2, H_Si, Al2O3=None, \
                      target_draws=100, verbose=False):
    """ N4 and Q3 expectation for a given composition M2O-B2O3-Al2O3-SiO2,
    where Al is assumed to preferentially consume modifier M, such that 
    only excess modifier ([M2O - Al2O3]) is partitioned between two 
    possibilities: tetrahedral boron conversion (N4), and NBO creation 
    on Si tetrahedra (Q3).

    Values of TG, M2O, B2O3, SiO2, and Al2O3 can be passed as scalars 
    for a single composition, or arrays for a composition series. TG 
    should have units of K and the others should represent mol fractions.

    The value of H_Si indicates the enthalpy associated with forming an
    NBO on a Si tetrahedron relative to forming a tetrahedral B unit.

    The composition will be scaled to an integer number of draws. The 
    number can be specified by providing an integer for target_draws, 
    or an array-like indicating a range can be provided. In the latter case, 
    the number within this range will be found that provides the best 
    approximation to the given composition (minimum rounding error).

    Note that using many draws can greatly increase calculation time.
    """

    N4 = []
    Q3 = []

    try:
        len(M2O)
        if Al2O3 is None:
            Al2O3 = np.zeros(len(M2O))
    except:
        if Al2O3 is None:
            Al2O3 = 0
        TG, M2O, B2O3, SiO2, Al2O3 = [np.array([item]) 
                                      for item in [TG, M2O, B2O3, SiO2, Al2O3]]

    # Two-state model
    M2O = np.maximum(M2O - Al2O3, 0)

    for i, m2o in enumerate(M2O):
        gSi = SiO2[i]
        gB = 2*B2O3[i]
        gAl = 2*Al2O3[i]
        m = 2*m2o

        if m <= 0 or gB == 0:
            N4.append(0)
            Q3.append(m/gSi)
        else:
            Tg = TG[i]
            wSi = np.exp(-H_Si/kB/Tg)

            # if range of target draws:
            try:
                draws_range = [draws for draws in range(
                               target_draws[0], target_draws[-1] + 1)]
                ssqs = []
                for draws in draws_range:
                    s = draws/m

                    residuals = np.array([round(element*s)/(element*s) 
                        for element in [gSi, gB, gAl, m] if element != 0])
                    ssq = chi_sq(np.ones(len(residuals)), residuals)
                    ssqs.append(ssq)

                if verbose:
                    print 'Draws:      ', draws_range
                    print 'SSQs(*100): ', [str(ssq)[0:6] for ssq in ssqs]
                min_ssq = np.min(ssqs)
                min_index = ssqs.index(min_ssq)
                adjusted_draws = draws_range[min_index]
                if verbose:
                    print 'Min ssq: {} at {} draws\n'.format(
                            min_ssq, adjusted_draws)
            except:
                traceback.print_exc()
                adjusted_draws = target_draws

            # obtain scaled composition
            s = adjusted_draws/m
            gSi = round(gSi*s)
            gB = round(gB*s)
            gAl = round(gAl*s)
            gNa = m = round(m*s)

            P = P_distribution(m, (gB, gSi), (1, wSi))

            nB, nSi = expectation(m, P)
            N4.append(nB/gB)
            if gSi == 0:
                Q3.append(0)
            else:
                Q3.append(nSi/gSi)

    N4, Q3 = [np.array(A) for A in (N4, Q3)]
    if verbose:
        print 'N4:\n{}'.format(N4)
        print 'Q3:\n{}'.format(Q3)

    if len(N4) > 1:
        return N4, Q3
    else:
        return N4[0], Q3[0]




def N4_L4_Q3_expectation(TG, M2O, B2O3, SiO2, Al2O3, H_Si, H_Al, 
                         target_draws=(20,40), verbose=False):
    """ N4, L4, and Q3 expectation for a given composition 
    M2O-B2O3-Al2O3-SiO2, where modifier is partitioned between three 
    possibilities: tetrahedral boron conversion (N4), tetrahedral Al 
    conversion (L4), and NBO creation on Si tetrahedra (Q3).

    Values of TG, M2O, B2O3, SiO2, and Al2O3 can be passed as scalars 
    for a single composition, or arrays for a composition series. TG 
    should have units of K and the others should represent mol fractions.

    Values of H_Si and H_Al indicate the enthalpy associated with 
    forming an NBO on a Si tetrahedron or forming a tetrahedral Al 
    unit, relative to forming a tetrahedral B unit.

    The composition will be scaled to an integer number of draws. The 
    number can be specified by providing an integer for target_draws, 
    or an array-like indicating a range can be provided. In the latter case, 
    the number within this range will be found that provides the best 
    approximation to the given composition (minimum rounding error).

    Note that using many draws can greatly increase calculation time.
    """

    N4 = []
    all_gB = []
    L4 = []
    all_gAl = []
    Q3 = []
    all_gSi = []

    # Three-state model with Al

    if verbose:
        print '\nBeginning data series'

    try:
        len(M2O)
    except:
        TG, M2O, B2O3, SiO2, Al2O3 = [np.array([item]) 
                                      for item in [TG, M2O, B2O3, SiO2, Al2O3]]

    for i, m2o in enumerate(M2O):
        gSi = SiO2[i]
        gB = 2*B2O3[i]
        gAl = 2*Al2O3[i]
        m = 2*m2o

        Tg = TG[i]
        wSi = np.exp(-H_Si/kB/Tg)
        wAl = np.exp(-H_Al/kB/Tg)

        try:
            draws_range = [draws for draws in range(
                           target_draws[0], target_draws[-1] + 1)]
            ssqs = []
            for draws in draws_range:
                s = draws/m

                residuals = np.array([round(element*s)/(element*s) 
                             for element in [gSi, gB, gAl, m] if element != 0])

                ssq = chi_sq(np.ones(len(residuals)), residuals)
                ssqs.append(ssq)

            if verbose:
                print 'Draws:      ', draws_range
                print 'SSQs(*100): ', [str(ssq)[0:6] for ssq in ssqs]
            min_ssq = np.min(ssqs)
            min_index = ssqs.index(min_ssq)
            adjusted_draws = draws_range[min_index]
            if verbose:
                print 'Min ssq: {} at {} draws\n'.format(
                        min_ssq, adjusted_draws)
        except:
            traceback.print_exc()
            adjusted_draws = target_draws

        s = adjusted_draws/m
        gSi = round(gSi*s)
        gB = round(gB*s)
        gAl = round(gAl*s)
        m = round(m*s)

        P = P_distribution(m, (gB, gSi, gAl), (1, wSi, wAl))

        nB, nSi, nAl = expectation(m, P)

        # prevent divide-by-zero errors
        if gB == 0:
            N4.append(1)
        else:
            N4.append(nB/gB)
        if gAl == 0:
            L4.append(1)
        else:
            L4.append(nAl/gAl)
        if gSi == 0:
            Q3.append(0)
        else:
            Q3.append(nSi/gSi)

        all_gB.append(gB)
        all_gAl.append(gAl)
        all_gSi.append(gSi)

    N4, L4, all_gB, all_gAl = [np.array(A) for A in (N4, L4, all_gB, all_gAl)]
    if verbose:
        print 'N4:\n{}'.format(N4)
        print 'gB:\n{}'.format(all_gB)
        print 'nB:\n{}'.format(N4*all_gB)
        print 'L4:\n{}'.format(L4)
        print 'gAl:\n{}'.format(all_gAl)
        print 'nAl:\n{}\n'.format(L4*all_gAl)

    if len(N4) > 1:
        return N4, L4, Q3
    else:
        return N4[0], L4[0], Q3[0]






def N4_N2_Q3_expectation(TG, M2O, B2O3, SiO2, Al2O3, H_Si, H_B2, 
                         target_draws=(20,40), verbose=False):
    """ N4, N2, and Q3 expectation for a given composition 
    M2O-B2O3-Al2O3-SiO2, where Al is assumed to preferentially 
    consume modifier M, such that only excess modifier ([M2O - Al2O3]) 
    is partitioned between three possibilities: tetrahedral boron 
    conversion (N4), NBO creation on trigonal boron (N2), and NBO 
    creation on Si tetrahedra (Q3).

    Values of TG, M2O, B2O3, SiO2, and Al2O3 can be passed as scalars 
    for a single composition, or arrays for a composition series. TG 
    should have units of K and the others should represent mol fractions.

    Values of H_Si and H_B2 indicate the enthalpy associated with 
    forming an NBO on a Si tetrahedron or on a trigonal B, relative 
    to forming a tetrahedral B unit.

    The composition will be scaled to an integer number of draws. 
    The number can be specified by providing an integer for target_draws, 
    or an array-like indicating a range can be provided. In the latter case, 
    the number within this range will be found that provides the best 
    approximation to the given composition (minimum rounding error).

    Note that using many draws can greatly increase calculation time.
    """

    N4 = []
    N2 = []
    all_gB = []
    Q3 = []
    all_gSi = []

    # Three-state model with B-NBO

    if verbose:
        print '\nBeginning data series'

    try:
        len(M2O)
    except:
        TG, M2O, B2O3, SiO2, Al2O3 = [np.array([item]) 
                                      for item in [TG, M2O, B2O3, SiO2, Al2O3]]

    M2O = np.maximum(M2O - Al2O3, 0)

    for i, m2o in enumerate(M2O):
        gSi = SiO2[i]
        gB = 2*B2O3[i]
        gAl = 2*Al2O3[i]
        m = 2*m2o

        if m <= 0 or gB == 0:
            N4.append(0)
            N2.append(0)
            Q3.append(m/gSi)
        else:
            Tg = TG[i]
            wSi = np.exp(-H_Si/kB/Tg)
            wB2 = np.exp(-H_B2/kB/Tg)

            try:
                draws_range = [draws for draws in range(
                                        target_draws[0], target_draws[-1] + 1)]
                ssqs = []
                for draws in draws_range:
                    s = draws/m
                    residuals = np.array([round(element*s)/(element*s) 
                        for element in [gSi, gB, gAl, m] if element != 0])
                    ssq = chi_sq(np.ones(len(residuals)), residuals)
                    ssqs.append(ssq)

                if verbose:
                    print 'Draws:      ', draws_range
                    print 'SSQs(*100): ', [str(ssq)[0:6] for ssq in ssqs]
                min_ssq = np.min(ssqs)
                min_index = ssqs.index(min_ssq)
                adjusted_draws = draws_range[min_index]
                if verbose:
                    print 'Min ssq: {} at {} draws\n'.format(
                                    min_ssq, adjusted_draws)
            except:
                traceback.print_exc()
                adjusted_draws = target_draws[-1]

            s = adjusted_draws/m
            gSi = round(gSi*s)
            gB = round(gB*s)
            gAl = round(gAl*s)
            m = round(m*s)

            P = P_distribution(m, (gB, gB, gSi), (1, wB2, wSi), 
                               same_type=(0,1))

            nB4, nB2, nSi = expectation(m, P)

            N4.append(nB4/gB)
            N2.append(nB2/gB)
            if gSi == 0:
                Q3.append(0)
            else:
                Q3.append(nSi/gSi)

            all_gB.append(gB)
            all_gSi.append(gSi)

    N4, N2, all_gB = [np.array(A) for A in (N4, N2, all_gB)]
    if verbose:
        print 'N4:\n{}'.format(N4)
        print 'gB:\n{}'.format(all_gB)
        print 'nB4:\n{}'.format(N4*all_gB)
        print 'N2:\n{}'.format(N2)
        print 'nB2:\n{}\n'.format(N2*all_gB)

    if len(N4) > 1:
        return N4, N2, Q3
    else:
        return N4[0], N2[0], Q3[0]