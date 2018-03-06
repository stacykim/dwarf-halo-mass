"""Driver script to loop over all halo mass histograms and run them through lensing_sn.py"""

import os
import subprocess
files = os.listdir('halo_mass_dist')

for file in files:
    if file != 'README.md':
        pref = file.rpartition('.')[0]
        subprocess.call(["python", "lensing_sn.py", 'halo_mass_dist/'+file, 'plots/'+pref])

