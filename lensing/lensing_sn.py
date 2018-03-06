import numpy as np
import matplotlib.pyplot as plt
import galsim

#### Basic inputs, should be made settable on the command line, but fix them here for now.
# Should we make and show diagnostic plots?
do_plot = True
# We are using a redshift slice, within which abundance and stellar vs. halo mass relation is
# assumed not to evolve.
z_min = 0.2
z_max = 0.4
# This is not the real effective redshift, but this detail doesn't matter for rough estimation of
# detectability.
z_eff = 0.5*(z_min + z_max)
# Pick an effective source redshift.  For now, we're not integrating over the source redshift
# distribution, just choosing a reasonable mean and normalization.
z_s = 0.9
# Noise-related stuff: shape noise, number density
n_eff = 26.0 # per arcmin^2
sigma_gamma = 0.25 # per component
# Cosmological parameters to be used for distance and volume calculations.
omega_m = 0.3
omega_lam = 1.-omega_m
# LSST sky fraction.
f_sky = 16000.0/40000
# Input file: expected format is three columns, with the first being the log10(M*h/Msun), the second
# being log10(comoving abundance in (h/Mpc)^3), and third is concentration.  We assume equal spacing
# in log10(M*h/Msun).  This has presumably be set by assuming some halo mass function, stellar
# vs. halo mass relation with mean and scatter, and then a lower and upper stellar mass limit.  All
# halo masses are assumed to be 200*rhocrit because that's what GalSim wants.
input_data_file = 'fake_input_data.txt'
# Establish radial bins for lensing signal, in kpc/h.
rp_min = 30.0
rp_max = 500.0
n_rp = 10

#### Read in data.
dat = np.loadtxt(input_data_file).transpose()
logm = dat[0,:]
abundance = dat[1,:]
conc = dat[2,:]
n_bin = len(logm)
print 'Read in %d mass bins from file %s'%(n_bin, input_data_file)

#### First calculation: Establish number of actual lenses in each bin in log(M).
# Plot the result for posterity.
cosmo = galsim.Cosmology(omega_m=omega_m, omega_lam=omega_lam)
# Get angular diameter distances for min, max redshifts.  The factor of 3000 is c/H0, which gives
# distances in Mpc/h.
da_min = 3000.0*cosmo.Da(z_min)
da_max = 3000.0*cosmo.Da(z_max)
# Get transverse comoving distances for volume calculation.
dm_min = da_min/(1.+z_min)
dm_max = da_max/(1.+z_max)
# Volume: use sky fraction for LSST and (4pi/3)*dm^3
vol_max = (4.*np.pi/3.)*f_sky*dm_max**3
vol_min = (4.*np.pi/3.)*f_sky*dm_min**3
total_volume = vol_max - vol_min # This volume is now in units of (Mpc/h)^3
n_lens = abundance*total_volume
if do_plot:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(logm, n_lens)
    ax.set_yscale('log')
    plt.xlabel(r'log10($M_{halo}$)')
    plt.ylabel(r'Number of lenses')
    plt.show()

#### Next calculation: stacked lensing signal.
# Set up bins in proper (not comoving) separation.  These are kpc/h.
rp_bin_edges = np.logspace(np.log10(rp_min), np.log10(rp_max), n_rp+1)
rp_bin_lower = rp_bin_edges[:-1]
rp_bin_upper = rp_bin_edges[1:]
rp_bin_mid = 0.5*(rp_bin_lower + rp_bin_upper)
# Convert to angular separations because for some reason GalSim wants this.
da_zeff = 3000.0*cosmo.Da(z_eff) # Mpc/h
ang_sep_rad = (rp_bin_mid/1000.0) / da_zeff
ang_sep_arcsec = galsim.radians/galsim.arcsec*ang_sep_rad
# Set up source positions: it wants 2D in the (x, y) plane around the lens.  So we use x=r, y=0.
input_pos = (ang_sep_arcsec, np.zeros_like(ang_sep_arcsec))

# Loop over mass bins to average the signal.
avg_shear = np.zeros(n_rp)
for i_bin in range(n_bin):
    # Define the NFWHalo
    nfw = galsim.NFWHalo(mass=10**logm[i_bin],
                         conc=conc[i_bin], redshift=z_eff,
                         cosmo=cosmo)
    # Select the tangential shear component given this input configuration
    shear_vals = nfw.getShear(pos=input_pos, z_s=z_s)
    gamma_t = np.sqrt(shear_vals[0]**2 + shear_vals[1]**2)
    avg_shear += n_lens[i_bin] * gamma_t
# Get appropriately number-weighted average signal across halo mass bins.
avg_shear /= np.sum(n_lens)
if do_plot:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(rp_bin_mid, avg_shear)
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.xlabel(r'$r_p$ [kpc/h]')
    plt.ylabel(r'$\gamma_t$')
    plt.show()

#### Uncertainty in stacked lensing signal.
# Shape noise errors!
# Get the area of each annular bin, then find the number of sources in each one.
ang_sep_rad_max = (rp_bin_upper/1000.0) / da_zeff
ang_sep_rad_min = (rp_bin_lower/1000.0) / da_zeff
annulus_area_rad = np.pi * (ang_sep_rad_max**2 - ang_sep_rad_min**2)
annulus_area_arcmin = annulus_area_rad * (galsim.radians/galsim.arcmin)**2
n_src = annulus_area_arcmin * n_eff
# And number of lens-src pairs
n_lens_src = n_src * np.sum(n_lens)
shape_noise = sigma_gamma / np.sqrt(n_lens_src)
if do_plot:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(rp_bin_mid, shape_noise)
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.xlabel(r'$r_p$ [kpc/h]')
    plt.ylabel(r'$\sigma(\gamma_t)$')
    plt.show()

# Signal-to-noise!
signal_to_noise = avg_shear / shape_noise
if do_plot:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(rp_bin_mid, signal_to_noise)
    ax.set_xscale('log')
    plt.xlabel(r'$r_p$ [kpc/h]')
    plt.ylabel(r'S/N')
    plt.show()
