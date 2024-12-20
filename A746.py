import pymc as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import corner
import matplotlib.collections as mcol
from matplotlib.legend_handler import HandlerLineCollection, HandlerTuple
from matplotlib.lines import Line2D


def S(lamb, sigRM):
    """Base S function for depolarization models"""
    return 2 * (sigRM ** 2) * (lamb ** 4)

def IFD(lamb, p0, sigRM):
    """Internal Faraday Dispersion"""
    S_value = S(lamb, sigRM)
    return p0 * (1 - np.exp(-S_value)) / S_value

def EFD(lamb, p0, sigRM):
    """External Faraday Dispersion"""
    S_value = S(lamb, sigRM)
    return p0 * np.exp(-S_value)

def build_model_IFD_IFD(wavelength, pol_frac, pol_err, initial_values=None):
    """Build PyMC model for IFD + IFD"""
    with pm.Model() as model:
        # Uniform priors for all parameters with initial values
        p0_1 = pm.Uniform('p0_1', lower=0.0, upper=1, initval=initial_values[0] if initial_values else None)
        sigRM_1 = pm.Uniform('sigRM_1', lower=10.0, upper=50.0, initval=initial_values[1] if initial_values else None)
        p0_2 = pm.Uniform('p0_2', lower=0.0, upper=1.0, initval=initial_values[2] if initial_values else None)
        sigRM_2 = pm.Uniform('sigRM_2', lower=0.0, upper=20.0, initval=initial_values[3] if initial_values else None)
        
        # Model prediction
        term1 = IFD(wavelength, p0_1, sigRM_1)
        term2 = IFD(wavelength, IFD(wavelength, p0_1, sigRM_1), sigRM_2)
        term3 = IFD(wavelength, p0_2, sigRM_2)
        model_pred = term1 + term2 + term3
        
        # Likelihood
        pm.Normal('likelihood', mu=model_pred, sigma=pol_err, observed=pol_frac)
        
    return model

def build_model_EFD_IFD(wavelength, pol_frac, pol_err, initial_values=None):
    """Build PyMC model for EFD + IFD"""
    with pm.Model() as model:
        # Uniform priors for all parameters
        p0_1 = pm.Uniform('p0_1', lower=0.0, upper=1.0, initval=initial_values[0] if initial_values else None)
        sigRM_1 = pm.Uniform('sigRM_1', lower=10.0, upper=50.0, initval=initial_values[1] if initial_values else None)
        p0_2 = pm.Uniform('p0_2', lower=0.0, upper=1.0, initval=initial_values[2] if initial_values else None)
        sigRM_2 = pm.Uniform('sigRM_2', lower=0.0, upper=10.0, initval=initial_values[3] if initial_values else None)
        
        # Model prediction
        term1 = IFD(wavelength, p0_1, sigRM_1)
        term2 = EFD(wavelength, IFD(wavelength, p0_1, sigRM_1), sigRM_2)
        term3 = EFD(wavelength, p0_2, sigRM_2)
        model_pred = term1 + term2 + term3
        
        # Likelihood
        pm.Normal('likelihood', mu=model_pred, sigma=pol_err, observed=pol_frac)
        
    return model

def build_model_EFD_EFD(wavelength, pol_frac, pol_err, initial_values=None):
    """Build PyMC model for EFD + EFD"""
    with pm.Model() as model:
        # Uniform priors for all parameters
        p0_1 = pm.Uniform('p0_1', lower=0.0, upper=1.0, initval=initial_values[0] if initial_values else None)
        sigRM_1 = pm.Uniform('sigRM_1', lower=10.0, upper=50.0, initval=initial_values[1] if initial_values else None)
        p0_2 = pm.Uniform('p0_2', lower=0.0, upper=1.0, initval=initial_values[2] if initial_values else None)
        sigRM_2 = pm.Uniform('sigRM_2', lower=0.0, upper=10.0, initval=initial_values[3] if initial_values else None)
        
        # Model prediction
        term1 = EFD(wavelength, p0_1, sigRM_1)
        term2 = EFD(wavelength, EFD(wavelength, p0_1, sigRM_1), sigRM_2)
        term3 = EFD(wavelength, p0_2, sigRM_2)
        model_pred = term1 + term2 + term3
        
        # Likelihood
        pm.Normal('likelihood', mu=model_pred, sigma=pol_err, observed=pol_frac)
        
    return model

def run_mcmc(model, draws=3000, tune=2000, chains=8):
    """Run MCMC sampling for a given model"""
    with model:
        # Initialize the chain starting points using jitter
        start = pm.find_MAP()
        
        # Use NUTS sampler with higher target_accept and store log likelihood
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=0.995,  # Increased target accept to reduce divergences
            return_inferencedata=True,
            initvals=start,
            idata_kwargs={"log_likelihood": True}  # Store log likelihood
        )
    return trace

def plot_fit_results(wavelength, pol_frac, pol_err, traces, flux=None, flux_err=None, lambda_fit=None):
    """Plot the fit results for all three models and flux data"""
    if lambda_fit is None:
        lambda_fit = np.linspace(np.min(wavelength), np.max(wavelength), 100)
    
    # Set up figure with two subplots sharing x-axis
    fig, ax1 = plt.subplots(figsize=(14, 8))
    ax2 = ax1.twinx()
    
    # Colors and line styles for different components
    colors = {
        'IFD_IFD': '#1f77b4',  # Blue
        'EFD_IFD': '#2ca02c',  # Green
        'EFD_EFD': '#ff7f0e',  # Orange
        'pol1': '#d62728',     # Red
        'pol2': 'Blue',        
        'flux': '#9467bd'      # Purple
    }
    
    line_styles = {
        'IFD_IFD': '--',      # Dashed
        'EFD_IFD': '-.',      # Dash-dot
        'EFD_EFD': ':'        # Dotted
    }
    
    # Plot polarization data points
    lamb_regular = wavelength[:2]
    pol_regular = pol_frac[:2]
    lamb_special = wavelength[2:]
    pol_special = pol_frac[2:]
    
    pol_1 = ax1.errorbar(lamb_regular, pol_regular, yerr=0.05 * pol_regular, fmt='o', 
                color=colors['pol1'], capsize=5, markersize=12, markeredgewidth=2)
    pol_2 = ax1.errorbar(lamb_special, pol_special, yerr=0.05 * pol_special, fmt='o',
                color=colors['pol2'], capsize=5, markersize=12, markeredgewidth=2)
    
    # Plot model fits
    model_handles = []
    for model_name, trace in traces.items():
        p0_1_samples = trace.posterior['p0_1'].values.flatten()
        sigRM_1_samples = trace.posterior['sigRM_1'].values.flatten()
        p0_2_samples = trace.posterior['p0_2'].values.flatten()
        sigRM_2_samples = trace.posterior['sigRM_2'].values.flatten()
        
        predictions = np.zeros((len(p0_1_samples), len(lambda_fit)))
        
        for i in range(len(p0_1_samples)):
            if model_name == 'IFD_IFD':
                term1 = IFD(lambda_fit, p0_1_samples[i], sigRM_1_samples[i])
                term2 = IFD(lambda_fit, IFD(lambda_fit, p0_1_samples[i], sigRM_1_samples[i]), sigRM_2_samples[i])
                term3 = IFD(lambda_fit, p0_2_samples[i], sigRM_2_samples[i])
            elif model_name == 'EFD_IFD':
                term1 = IFD(lambda_fit, p0_1_samples[i], sigRM_1_samples[i])
                term2 = EFD(lambda_fit, IFD(lambda_fit, p0_1_samples[i], sigRM_1_samples[i]), sigRM_2_samples[i])
                term3 = EFD(lambda_fit, p0_2_samples[i], sigRM_2_samples[i])
            else:  # EFD_EFD
                term1 = EFD(lambda_fit, p0_1_samples[i], sigRM_1_samples[i])
                term2 = EFD(lambda_fit, EFD(lambda_fit, p0_1_samples[i], sigRM_1_samples[i]), sigRM_2_samples[i])
                term3 = EFD(lambda_fit, p0_2_samples[i], sigRM_2_samples[i])
            
            predictions[i] = term1 + term2 + term3
        
        pred_mean = np.percentile(predictions, 50, axis=0)
        pred_lower = np.percentile(predictions, 16, axis=0)
        pred_upper = np.percentile(predictions, 84, axis=0)
        
        handle, = ax1.plot(lambda_fit, pred_mean, linestyle=line_styles[model_name], 
                         color=colors[model_name], label=f'Fitted {model_name.replace("_", " + ")}',
                         linewidth=5)
        ax1.fill_between(lambda_fit, pred_lower, pred_upper, 
                        color=colors[model_name], alpha=0.2)
        model_handles.append(handle)
    
    # Plot flux data
    if flux is not None and flux_err is not None:
        flux_plot = ax2.errorbar(wavelength, flux * 1000, yerr=flux_err * 1000, 
                               fmt='s', color=colors['flux'], label='Flux Density (mJy)',
                               alpha=0.3, capsize=5, markersize=8, markeredgewidth=2)
    
    # Set labels with larger fonts
    ax1.set_xlabel('Wavelength [m]', fontsize=24)
    ax1.set_ylabel('Linear Fractional Polarization', fontsize=24)
    ax2.set_ylabel('Flux Density [mJy]', fontsize=24)
    ax1.set_title('Linear Fractional Polarization and Measured Fluxes in NW Relic', fontsize=26, y=1.05)
    
    # Create separate legends
    # Legend for data points
    legend1 = ax1.legend(handles=[(pol_1, pol_2), flux_plot],
                        labels=['Linear Fractional Polarization', 'Flux Density (mJy)'],
                        loc='upper right', bbox_to_anchor=(0.5, 1),
                        fontsize=18, scatterpoints=1, numpoints=1,
                        handler_map={tuple: HandlerTuple(ndivide=None)})
    
    # Legend for model fits
    legend2 = ax1.legend(handles=model_handles,
                        loc='center', bbox_to_anchor=(0.7, 0.9),
                        fontsize=16)
    
    # Add both legends
    ax1.add_artist(legend1)
    ax1.add_artist(legend2)
    
    # Configure tick parameters
    ax1.tick_params(axis='both', which='major', labelsize=22)
    ax2.tick_params(axis='both', which='major', labelsize=22)
    ax2.tick_params(which='major', length=10, width=2, direction='in')
    ax2.tick_params(which='minor', length=5, width=1, direction='in')
    ax2.minorticks_on()
    ax1.tick_params(which='major', length=10, width=2, direction='in')
    ax1.tick_params(which='minor', length=5, width=1, direction='in')
    ax1.minorticks_on()
    
    # Add grid
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig


def plot_corner(trace, model_name):
    """Create corner plot for model parameters"""
    # Extract samples
    samples = np.column_stack([
        trace.posterior['p0_1'].values.flatten(),
        trace.posterior['sigRM_1'].values.flatten(),
        trace.posterior['p0_2'].values.flatten(),
        trace.posterior['sigRM_2'].values.flatten()
    ])
    
    # Labels for parameters (fixed string literals)
    labels = [
        r'$p_{0,1}$',
        r'$\sigma_{RM,1}$',
        r'$p_{0,2}$',
        r'$\sigma_{RM,2}$'
    ]
    
    # Create corner plot
    fig = corner.corner(
        samples, 
        labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 16},
        label_kwargs={"fontsize": 16},
        title_fmt='.3f',
        figsize=(12, 12),
        hist_kwargs={'density': True},
        plot_datapoints=False,
        plot_density=True,
        fill_contours=True,
        levels=(0.68, 0.95),
        color='darkblue',
        bins=50
    )
    
    # Adjust plots
    axes = np.array(fig.axes).reshape((4, 4))
    for i in range(4):
        for j in range(4):
            if axes[i,j] is not None:
                axes[i,j].tick_params(labelsize=12)
                if i > j:  # Contour plots
                    axes[i,j].grid(True, alpha=0.3)
    
    # Add model name as suptitle
    fig.suptitle(f"{model_name}", fontsize=20, y=1.02)
    
    return fig

def compare_models(traces):
    """Compare models using WAIC and LOO-CV"""
    try:
        # Try LOO comparison first
        comparison = az.compare(traces, ic="loo", scale="deviance")
    except Exception as e:
        print("LOO comparison failed, falling back to WAIC")
        try:
            # Fall back to WAIC if LOO fails
            comparison = az.compare(traces, ic="waic", scale="deviance")
        except Exception as e:
            print(f"Model comparison failed: {str(e)}")
            comparison = None
    return comparison

# Example usage
wavelength = np.array([0.21, 0.18, 0.5093, 0.4771, 0.4499])
pol_frac = np.array([0.3884416, 0.53899263, 0.15964682, 0.18899837, 0.2118822])
pol_err = 0.15 * pol_frac

# Initial guesses from least squares fits
initial_IFD_IFD = [1.0, 56.82357216, 0.24194593, 2.42299692]
initial_EFD_IFD = [0.35961964, 3.77203634, 0.79903226, 29.53726422]
initial_EFD_EFD = [0.4231681, 25.56901652, 0.32939448, 2.31859455]

# Build and run models
print("Fitting IFD+IFD model...")
model_IFD_IFD = build_model_IFD_IFD(wavelength, pol_frac, pol_err)
trace_IFD_IFD = run_mcmc(model_IFD_IFD)

print("Fitting EFD+IFD model...")
model_EFD_IFD = build_model_EFD_IFD(wavelength, pol_frac, pol_err)
trace_EFD_IFD = run_mcmc(model_EFD_IFD)

print("Fitting EFD+EFD model...")
model_EFD_EFD = build_model_EFD_EFD(wavelength, pol_frac, pol_err)
trace_EFD_EFD = run_mcmc(model_EFD_EFD)

traces = {
    'IFD_IFD': trace_IFD_IFD,
    'EFD_IFD': trace_EFD_IFD,
    'EFD_EFD': trace_EFD_EFD
}


wavelength = np.array([0.21, 0.18, 0.5093, 0.4771, 0.4499,0.4242])
pol_frac = np.array([0.3884416 , 0.53899263, 0.15964682, 0.18899837, 0.2118822, 0.405 ])
pol_err = 0.15 * pol_frac
flux = np.array([0.00071989, 0.00050298, 0.00195806, 0.00149635, 0.00163378,
       0.00129418])
flux_err = 0.1*flux
fig_fit = plot_fit_results(wavelength, pol_frac, pol_err, traces, flux=flux, flux_err=flux_err, lambda_fit=None)
plt.savefig('model_fits.png', bbox_inches='tight', dpi=300)

# Create corner plots
fig_corner_ifd_ifd = plot_corner(trace_IFD_IFD, 'IFD + IFD')
plt.savefig('corner_IFD_IFD.png', bbox_inches='tight', dpi=300)

fig_corner_efd_ifd = plot_corner(trace_EFD_IFD, 'EFD + IFD')
plt.savefig('corner_EFD_IFD.png', bbox_inches='tight', dpi=300)

fig_corner_efd_efd = plot_corner(trace_EFD_EFD, 'EFD + EFD')
plt.savefig('corner_EFD_EFD.png', bbox_inches='tight', dpi=300)

# Model comparison and parameter summaries
comparison = compare_models(traces)
print("\nModel Comparison:")
print(comparison)

print("\nIFD + IFD Summary:")
print(az.summary(trace_IFD_IFD, var_names=['p0_1', 'sigRM_1', 'p0_2', 'sigRM_2']))

print("\nEFD + IFD Summary:")
print(az.summary(trace_EFD_IFD, var_names=['p0_1', 'sigRM_1', 'p0_2', 'sigRM_2']))

print("\nEFD + EFD Summary:")
print(az.summary(trace_EFD_EFD, var_names=['p0_1', 'sigRM_1', 'p0_2', 'sigRM_2']))
