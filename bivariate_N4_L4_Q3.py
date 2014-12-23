""" Fit literature N4 data vs. bivariate Wallenius model

Considers three possible outcomes for modifier partitioning:
    Trigonal B -> tetrahedral B (N4 = fraction tetrahedral B)
    Trigonal Al -> tetrahedral Al (L4 = fraction tetrahedral Al)
    NBO conversion on Si tetrahedron, Q4 -> Q3

"""

from scipy.optimize import curve_fit
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from hypergeometric import get_data, get_col, N4_L4_Q3_expectation, chi_sq

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
f = lambda x, H_Si, H_Al: N4_L4_Q3_expectation(
        Tg, Na2O, B2O3, SiO2, Al2O3, H_Si, H_Al,
        target_draws=(10, 30))

# since f returns N4, L4 and Q3, create lambda f_N4 that returns only N4
f_N4 = lambda x, H_Si, H_Al: f(x, H_Si, H_Al)[0]

print 'Beginning fit with bivariate model (this may take awhile...)'

# pass f_N4 to curve_fit
fit, cov = curve_fit(f_N4, X, N4, [0, 0])

# feed best fit result back into f to get best-fit N4, L4 and Q3
fitN4, fitL4, fitQ3 = f(X, fit[0], fit[1])

print "Wallenius fit yields H_Si={} and H_Al={} with ssq={}".format(
                                             fit[0], fit[1], chi_sq(N4, fitN4))

# Plot setup
fig = plt.figure()
fig.subplots_adjust(bottom=0.12, top=0.88)
ax = fig.add_subplot(111)
ax.plot(X, N4, label="$\mathrm{N}_4$ experiment", marker='s', 
        color='black', lw=1)
ax.plot(X, fitN4, label="$\mathrm{N}_4$ best-fit", marker='s', 
        color='b', lw=1)
ax.plot(X, np.minimum(1, Na2O/Al2O3), label="$\mathrm{L}_4$ expected", 
        marker='o', color='black', lw=1)
ax.plot(X, fitL4, label="$\mathrm{L}_4$ predicted", marker='o', 
        color='purple', lw=1)
ax.plot(X, fitQ3, label="$\mathrm{Q}_3$ predicted", marker='o', 
        color='green', lw=1)
textstr = '$\Delta$H$_{{^{{[3]}}\mathrm{{Si}}}}$ =' + \
            ' ${}$\n$\Delta$H$_{{^{{[4]}}\mathrm{{Al}}}}$ = ${}$'
textstr = textstr.format(str(round(fit[0]*10000)/10000)[:6], 
                         str(round(fit[1]*10000)/10000)[:7])
ax.text(0.70, 0.28, textstr, transform=ax.transAxes, fontsize=16,
        verticalalignment='top')

# Plot formatting
ax.set_xlabel(xlabel)
ax.set_ylim([0, 1])
ax.legend(loc=7)
ax.set_ylabel("Species fraction")
ax.set_title("Wallenius 3-state speciation ( $^{[4]}\mathrm{B}$, " + \
             "$^{[3]}\mathrm{Si}$, and ${}^{[4]}\mathrm{Al}$ )", y=1.03)
ax.title.set_fontsize(18)
for item in ([ax.xaxis.label, ax.yaxis.label]):
    item.set_fontsize(16)
for item in ax.get_xticklabels() + ax.get_yticklabels():
    item.set_fontsize(14)

plt.show()