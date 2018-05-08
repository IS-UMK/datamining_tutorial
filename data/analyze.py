import os
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
from sklearn import decomposition
import sklearn

N_WINDOW = 70 #parameter for
SERIA = 1 #seria == 0 - ppierwszy pli, 1- drugi plik


dirs = os.listdir('in_data')
dirs = sorted(dirs)

person2data = {}

for d in dirs:
    fns = os.listdir('./in_data/' + d)
    fns = sorted(fns)
    data_f = []
    for fn in fns:
        data = open('./in_data/' + d + '/'+ fn).readlines()
        rrs =  [int(l.strip()) for l in data]
        rr = np.array(rrs)
        d_rr = rr[1:] - rr[:-1]
        # m, s = np.mean(rrs), np.std(rrs)
        # rrs = [r for r in rrs if (m-r) < s]

        data_f.append(rr[1:][d_rr< 200])
    person2data[d] = data_f

fig = plt.figure(figsize = (30,10))
# axes = {}
p_ids = sorted(person2data.keys())

ax_rr_art = fig.add_subplot(611); plt.title('RR')
ax_rr = fig.add_subplot(612, sharex = ax_rr_art, sharey = ax_rr_art)


ax_sdnn_art = fig.add_subplot(613, sharex = ax_rr_art,); plt.title('SDNN')
ax_sdnn = fig.add_subplot(614, sharex = ax_rr_art, sharey = ax_sdnn_art)

ax_sd1sd2_art = fig.add_subplot(615, sharex = ax_rr_art,); plt.title('SD1/SD2')
ax_sd1sd2 = fig.add_subplot(616,sharex = ax_rr_art, sharey = ax_sd1sd2_art)

#plot RR
rr_a = np.array(person2data['ARTYSTA'][SERIA])
ax_rr_art.plot(rr_a, 'r-')

for subject, data in person2data.iteritems():
    if subject == 'ARTYSTA' : continue
    rr = np.array(data)[SERIA]
    ax_rr.plot(rr)


# plot sdnn

rr = np.array(person2data['ARTYSTA'])[SERIA]
d_rr = rr[1:] - rr[:-1]
sdnn = []
for n in range(len(d_rr) - N_WINDOW):
    sdnn_i = np.std(  d_rr[n:n+N_WINDOW] )
    sdnn.append(sdnn_i)
ax_sdnn_art.plot(sdnn, 'r-')

for subject, data in person2data.iteritems():
    if subject == 'ARTYSTA' : continue
    rr = np.array(data[SERIA])
    d_rr = rr[1:] - rr[:-1]
    sdnn = []
    for n in range(len(d_rr) - N_WINDOW):
        sdnn_i = np.std(  d_rr[n:n+N_WINDOW] )
        sdnn.append(sdnn_i)
    ax_sdnn.plot(sdnn)


# PLOT POINCARE

def dummy_poincare(rr):
    fi = np.deg2rad(45)
    M = np.array([[np.cos(fi), -np.sin(fi)], [np.sin(fi), np.cos(fi)]])

    P = np.vstack([rr[1:], rr[:-1]]).T
    rr_r  = np.apply_along_axis(M.dot, 1, P)
    sd1 = np.std(rr_r[:,0])
    sd2 = np.std(rr_r[:,1])
    return sd1, sd2


rr = np.array(person2data['ARTYSTA'])[SERIA]
d_rr = rr[1:] - rr[:-1]
sd1sd2 = []
for n in range(len(d_rr) - N_WINDOW):
    rr = d_rr[n:n+N_WINDOW]
    sd1, sd2 = dummy_poincare(rr)
    sd1sd2.append(float(sd1)/sd2)

ax_sd1sd2_art.plot(np.array(sd1sd2), 'r-')

for subject, data in person2data.iteritems():
    if subject == 'ARTYSTA' : continue
    rr = np.array(data[SERIA])
    d_rr = rr[1:] - rr[:-1]
    sd1sd2 = []
    for n in range(len(d_rr) - N_WINDOW):
        rr = d_rr[n:n+N_WINDOW]
        sd1, sd2 = dummy_poincare(rr)
        sd1sd2.append(float(sd1)/sd2)

    ax_sd1sd2.plot(np.array(sd1sd2))

plt.tight_layout()
plt.show()
