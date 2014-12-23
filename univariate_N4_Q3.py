""" Fit literature N4 data vs. univariate Wallenius model

Considers two possible outcomes for modifier partitioning:
    Trigonal B -> tetrahedral B (N4 = fraction tetrahedral B)
    NBO conversion on Si tetrahedron, Q4 -> Q3

When present, Al2O3 is assumed to preferentially consume available modifier
such that only the remainder will be partitioned.

"""

from scipy.optimize import curve_fit
import matplotlib
import matplotlib.pyplot as plt

from hypergeometric import get_data, get_col, N4_Q3_expectation, chi_sq

font = { 'family': 'Arial' }
matplotlib.rc('font', **font)

# get Zheng2012 data
data = get_data('NaBAlSi_data_Zheng2012.csv')
SiO2, Al2O3, B2O3, Na2O = [get_col(data, i) for i in range(4)]
Tg = get_col(data, 4)
N4 = get_col(data, 5)/100
X = Al2O3    # plot vs Al2O3
xlabel = "[$\mathrm{Al}_2\mathrm{O}_3$]"




#-- Fit N4 calculation to data using scipy's curve_fit --#

# curve_fit expects a function of X and fitting params that returns N4, so...

# wrap expectation function in a lambda with args of only X and fitting params
f = lambda X, H_Si: N4_Q3_expectation(Tg, Na2O, B2O3, SiO2, H_Si, Al2O3,
                                      target_draws=(20, 40))

# since f returns N4 and Q3, create lambda f_N4 that returns only N4
f_N4 = lambda X, H_Si: f(X, H_Si)[0]

# pass f_N4 to curve_fit
fit, cov = curve_fit(f_N4, X, N4, [0])

# feed best fit result back into f to get best-fit N4 and Q3
fitN4, fitQ3 = f(X, fit[0])

print "Wallenius fit yields H={} with ssq={}".format(fit[0], chi_sq(N4, fitN4))

# plot setup
fig = plt.figure()
fig.subplots_adjust(bottom=0.12, top=0.88)
ax = fig.add_subplot(111)

ax.plot(X, N4, label="Experiment", marker='s', color='black', lw=1)
ax.plot(X, fitN4, label="$\mathrm{N}_4$ best-fit", marker='s', color='b', lw=1)
ax.plot(X, fitQ3, label="$\mathrm{Q}_3$ predicted",
        marker='o', color='green', lw=1)

# Plot formatting

ax.set_xlabel(xlabel)
ax.legend(loc=1)
ax.set_ylabel("Species fraction")
ax.set_title("Wallenius 2-state speciation " +\
             "( $^{[4]}\mathrm{B}$ and $^{[3]}\mathrm{Si}$ )", y=1.03)
ax.title.set_fontsize(18)
for item in ([ax.xaxis.label, ax.yaxis.label]):
    item.set_fontsize(16)
for item in ax.get_xticklabels() + ax.get_yticklabels():
    item.set_fontsize(14)

plt.show()