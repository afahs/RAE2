from astropy.coordinates import angular_separation
from astropy.coordinates import SkyCoord, get_sun,FK4,get_body, EarthLocation
from astropy.time import Time

from datetime import timedelta
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import os
import glob
import numpy as np
import astropy.units as u
from scipy.interpolate import splrep,BSpline
from matplotlib.ticker import MaxNLocator
from tqdm.notebook import tqdm

def bandToFreq(band):
    # Convert the input to a numpy array (if it's not already)
    band = np.asarray(band)
    
    # Create a mapping of band numbers to frequency values
    band_to_freq = {
        1: 0.45, 2: 0.70, 3: 0.90, 4: 1.31,
        5: 2.20, 6: 3.93, 7: 4.70, 8: 6.55, 9: 9.18
    }

    # Use np.vectorize to apply the mapping to each element in the array
    freq = np.vectorize(lambda b: band_to_freq.get(b, -1))(band)
    
    return freq


def raeAngFromSource(locations, source):
    position_x = np.array(locations['position_x'], dtype=float)
    position_y = np.array(locations['position_y'], dtype=float)
    position_z = np.array(locations['position_z'], dtype=float)
    
    ra,dec = cartToSphere(position_x,position_y,position_z)
    
    
    #ra = (long + 180) % 360 * u.deg
    #dec = -lat * u.deg
    ra = ra*u.deg
    dec = dec*u.deg
    
    raeCoord = SkyCoord(ra, dec, frame='icrs')
    sourceCoord = SkyCoord(ra=source[0], dec=source[1], frame='icrs',unit=(u.hourangle, u.deg))

    ang_sep = raeCoord.separation(sourceCoord)
    
    diff = np.diff(ang_sep)
    signs = np.sign(diff)
    signs = np.append(signs,signs[-1])#need to find better way to add signs
    for i in range(1, len(signs)):
        if signs[i] == 0:
            signs[i] = signs[i - 1]
    

    return signs*ang_sep.value


def angularSepEarth(data):#this doesnt match ang_sep 2 on where it flips
    posList = ['x','y','z']
    earthMoonDist = 384000#km (avg distance)
    earthUnitVector = np.zeros((len(data),len(posList)))
    pos = np.zeros((len(data),len(posList)))
    for i in range(0,len(data)):
        for j in range(0,len(posList)):
            earthUnitVector[i][j] = data.iloc[i]['earth_unit_vector_'+posList[j]]#pointing from center of moon
            pos[i][j] = data.iloc[i]['position_'+posList[j]]
    earthVector = earthUnitVector*earthMoonDist
    raeVector = -earthVector+pos#should be earth centered position of rae
    
    posUnitVector = -pos/np.linalg.norm(pos, axis=1)[:, np.newaxis]#vector from RAE2 to moon
    raeUnitVector = -raeVector/np.linalg.norm(raeVector, axis=1)[:, np.newaxis]#vector from RAE to earth
    moonUnitVector = -earthUnitVector#unit vector from earth to moon
    
    latEarth, lonEarth = cartToSphere(raeUnitVector[:, 0], raeUnitVector[:, 1], raeUnitVector[:, 2])#from RAE to Earth
    latPos, lonPos = cartToSphere(posUnitVector[:, 0], posUnitVector[:, 1], posUnitVector[:, 2])#RAE to moon
    latME, lonME = cartToSphere(earthUnitVector[:,0],earthUnitVector[:,1],earthUnitVector[:,2])#moon to earth
    latMR, lonMR = cartToSphere(pos[:, 0], pos[:, 1], pos[:, 2])#moon to RAE
    
    
    coord1 = SkyCoord(ra = lonEarth, dec = latEarth, frame='icrs', unit = (u.deg,u.deg))
    coord2 = SkyCoord(ra = lonPos, dec = latPos, frame='icrs', unit = (u.deg,u.deg))
    coord3 = SkyCoord(ra = lonME, dec = latME, frame='icrs', unit = (u.deg,u.deg))
    coord4 = SkyCoord(ra = lonMR, dec = latMR, frame='icrs', unit = (u.deg,u.deg))
    
    pos_ang = coord3.position_angle(coord4).deg
    angle = coord1.separation(coord2).deg
    
    angle = np.where(pos_ang > 180, angle, -angle)
    return angle


def angularSepEarth2(data):
    posList = ['x','y','z']
    earthMoonDist = 384000#km (avg distance)
    earthUnitVector = np.zeros((len(data),len(posList)))
    pos = np.zeros((len(data),len(posList)))
    for i in range(0,len(data)):
        for j in range(0,len(posList)):
            earthUnitVector[i][j] = data.iloc[i]['earth_unit_vector_'+posList[j]]#pointing from center of moon
            pos[i][j] = data.iloc[i]['position_'+posList[j]]
    earthVector = earthUnitVector*earthMoonDist
    raeVector = -earthVector+pos#should be earth centered position of rae
    
    
    dotProd = np.sum(raeVector * pos, axis=1) # Row-wise dot product

    norms = np.linalg.norm(raeVector, axis=1) * np.linalg.norm(pos, axis=1)
    cosTheta = np.clip(dotProd / norms,-1.0,1.0)
    angle = np.arccos(cosTheta)  # Angle in radians
    
    diff = np.diff(np.sum(raeVector*raeVector,axis=1))
    sign = np.sign(diff)
    sign = np.append(sign,sign[-1])
    for i in range(1, len(sign)):
        if sign[i] == 0:
            sign[i] = sign[i - 1]
    return sign*angle*180/np.pi  # Convert to degrees


# In[6]:


def cartToSphere(x, y, z):
    # Calculate radius
    r = np.sqrt(x**2 + y**2 + z**2)
    
    # Calculate declination (latitude in spherical terms)
    dec = np.arcsin(z / r)
    
    # Calculate right ascension (longitude in spherical terms)
    ra = np.arctan2(y, x)
    
    # Convert to degrees
    dec = np.degrees(dec)
    ra = np.degrees(ra)
    
    # Ensure ra is in the range [0, 360]
    ra = ra % 360
    
    return ra, dec


# In[7]:


def isVisible(data,angle):
    position_x = np.array(data['position_x'], dtype=float)
    position_y = np.array(data['position_y'], dtype=float)
    position_z = np.array(data['position_z'], dtype=float)
    
    # Calculate spherical coordinates for all locations
    moonDist = np.sqrt(position_x**2 + position_y**2 + position_z**2)
    moonRad = 1740 #km
    moonWidth = np.degrees(np.tan(moonRad/moonDist))
    isVis = np.abs(angle) > moonWidth

    # Use np.where to apply the conditions over arrays
    
    return isVis


# In[8]:


def solarSystemAngles(data,obj):#I know about the 2 week vs 1 week thing, the code was fastr than I thought
    angles = []

    # Process data in 2-week chunks
    two_weeks = pd.Timedelta(weeks=1)
    start_time = data.index.min()
    end_time = data.index.max()

    current_time = start_time
    while current_time <= end_time:
        # Define the chunk
        chunk_mask = (data.index >= current_time) & (data.index < current_time + two_weeks)
        chunk = data.loc[chunk_mask]
        
        if chunk.empty:
            current_time += two_weeks
            continue
        
        # Calculate the midpoint time for querying the solar system position
        mid_time = Time(chunk.index[len(chunk) // 2].to_pydatetime())
        
        # Query the source position for the midpoint time
        if obj.lower() == 'sun':
            source_coord = get_sun(mid_time).transform_to(FK4(equinox=Time('B1950')))
        else:
            source_coord = get_body(obj.lower(), mid_time).transform_to(FK4(equinox=Time('B1950')))
        
        # Convert the source position to RA and Dec
        source_ra = source_coord.ra.deg  # RA in degrees
        source_dec = source_coord.dec.deg  # Dec in degrees
        source_pos = [source_ra, source_dec]
        
        # Calculate angles for the entire chunk
        # Ensure raeAngFromSource returns a list/array of the same length as the chunk
        chunk_angles = np.array(raeAngFromSource(chunk, source_pos))  # Ensure output is a numpy array
        
        # Check if the output has the same length as the chunk
        if len(chunk_angles) != len(chunk):
            raise ValueError(f"Length mismatch: chunk length is {len(chunk)}, but output length is {len(chunk_angles)}")
        
        # Store the results (make sure angles is being updated correctly)
        angles.append(chunk_angles)
        
        # Move to the next 2-week period
        current_time += two_weeks

    # Return the angles as a concatenated list or array
    return np.concatenate(angles)


# In[9]:


def dictAdd(dic1,dic2):
    dic = dic1.copy()
    for key,value in dic2.items():
        if key in dic:
            dic[key]+=value
        else:
            dic[key] = value
    return dic


# In[10]:


def occulted(data,col = 'isVis'):
    numOfOrbits = {}
    orbitDelta = {}
    for i in range(1,len(data)):
        
        if (data[col][i-1] ==True and data[col][i]==False):
            occultTime = data.index[i]
            occultTimeStart = occultTime - pd.Timedelta(minutes = 10)
            occultTimeEnd = occultTime +pd.Timedelta(minutes = 10)
            occulted = data[(data.index >= occultTime) & (data.index <= occultTimeEnd)].copy()
            notOcculted = data[(data.index >= occultTimeStart) & (data.index <= occultTime)].copy()
            comparePowerResult,N = comparePower(occulted,notOcculted)
            numOfOrbits = dictAdd(numOfOrbits,N)
            orbitDelta = dictAdd(orbitDelta,comparePowerResult)
    return orbitDelta,numOfOrbits


# In[11]:


def sigmaClip(arr, n=5, max_iter=10):
    """Perform iterative sigma clipping to remove outliers."""
    arr = np.asarray(arr)  # Ensure input is a NumPy array
    for _ in range(max_iter):
        med = np.median(arr)
        sig = np.std(arr)
        filtered = arr[(arr > med - n * sig) & (arr < med + n * sig)]
        
        # Stop if no more values are clipped
        if len(filtered) == len(arr):
            break
        arr = filtered

    return arr if arr.size > 0 else np.array([med])


# In[12]:


def percentileClip(arr, perc = 5, iters = 1):
    arr = np.asarray(arr)
    for _ in range(iters):
        lower_bound = np.percentile(arr, perc)
        upper_bound = np.percentile(arr, 100 - perc)
        arr = arr[(arr >= lower_bound) & (arr <= upper_bound)]  # Keep only values within bounds
    return arr


# In[13]:


def occultedHistogramNaive(data,col='isVis'):
    freqs = data['frequency_band'].unique()
    histograms = {}
    for freq in freqs:
        freq_data = data[data['frequency_band']==freq]
        occulted = freq_data[freq_data[col]==False]['rv1_coarse']
        nonOcculted = freq_data[freq_data[col]==True]['rv1_coarse']
        occulted_clip = sigmaClip(occulted,n=1)
        nonOcculted_clip = sigmaClip(nonOcculted,n=1)
        for i in range(0,3):
            occulted_clip = sigmaClip(occulted_clip,n=3)
            nonOcculted_clip = sigmaClip(nonOcculted_clip,n=3)
        occulted_hist, occulted_bins = np.histogram(occulted_clip, bins='auto')
        nonOcculted_hist, nonOcculted_bins = np.histogram(nonOcculted_clip, bins='auto')
        histograms[freq] = {
            'occulted': {'hist': occulted_hist, 'bins': occulted_bins},
            'nonOcculted': {'hist': nonOcculted_hist, 'bins': nonOcculted_bins}
        }
    # Return the histogram counts and bins as a dictionary
    return histograms


# In[14]:


def occultationStatistics(data, col='isVis', window=pd.Timedelta(minutes=10), perc=5):
    freqs = data['frequency_band'].unique()
    stats = {}

    # Identify occultation periods
    occultation_changes = data[col].astype(int).diff()  # Detect changes in isVis
    start_times = data.index[(occultation_changes == -1)]  # False → True (start of occultation)
    end_times = data.index[(occultation_changes == 1)]  # True → False (end of occultation)

    # Ensure start_times and end_times align properly
    if len(end_times) > 0 and (len(start_times) == 0 or start_times[0] > end_times[0]):
        end_times = end_times[1:]  # Remove first end_time if it comes before any start_time

    if len(start_times) > len(end_times):
        start_times = start_times[:-1]  # Remove extra start_time if unmatched

    # Filter occultation periods that are at least 2 minutes long
    valid_pairs = [(s, e) for s, e in zip(start_times, end_times) if (e - s) >= pd.Timedelta(minutes=2)]
    
    # **Fix for empty valid_pairs**  
    if valid_pairs:
        start_times, end_times = zip(*valid_pairs)
    else:
        start_times, end_times = [], []

    # Loop over frequency bands
    for freq in tqdm(freqs, desc="Processing frequencies"):
        freq_data = data[data['frequency_band'] == freq]
        occultation_stats = {'median': [], 'std': []}
        non_occultation_stats = {'median': [], 'std': []}

        for start, end in zip(start_times, end_times):
            # Select occulted region
            occultation_period = freq_data.loc[(freq_data.index >= start) & (freq_data.index <= end)]

            # Select non-occulted regions before and after
            pre_occult = freq_data.loc[(freq_data.index >= start - window) & 
                                       (freq_data.index < start) & 
                                       (freq_data[col] == True)]
            
            post_occult = freq_data.loc[(freq_data.index > end) & 
                                        (freq_data.index <= end + window) & 
                                        (freq_data[col] == True)]
            
            non_occult = pd.concat([pre_occult, post_occult])  # Combine both non-occultation regions
            
            if occultation_period.empty:
                print(f"Skipping empty occultation period: {start} to {end}")
                continue
            if non_occult.empty:
                print(f"Skipping empty non occult: {start} to {end}")
                continue
            # Extract signal values
            occultation_period_sig = occultation_period['rv1_coarse']
            non_occult_sig = non_occult['rv1_coarse']
            #occultation_period_sig = percentileClip(occultation_period_sig, perc=perc)
            #non_occult_sig = percentileClip(non_occult_sig, perc=perc)  
            occultation_period_sig = sigmaClip(occultation_period_sig,n=5)
            non_occult_sig = sigmaClip(non_occult_sig,n=5)
            # **Fix: Skip if percentileClip removes all elements**
            if len(occultation_period_sig) == 0 or len(non_occult_sig) == 0:
                print(f"Skipping due to empty clipped data: {start} to {end}")
                continue

            # Compute statistics
            occultation_stats['median'].append(np.nanmedian(occultation_period_sig))
            occultation_stats['std'].append(np.nanstd(occultation_period_sig))

            non_occultation_stats['median'].append(np.nanmedian(non_occult_sig))
            non_occultation_stats['std'].append(np.nanstd(non_occult_sig))
        
        # Store statistics for this frequency band
        stats[freq] = {
            'occulted': occultation_stats,
            'non_occulted': non_occultation_stats
        }

    return stats


# In[15]:


def occultationStatisticsSigma(data, col='isVis', window=pd.Timedelta(minutes=10),n=5,antenn = 'rv1_coarse'):
    freqs = data['frequency_band'].unique()
    stats = {}

    # Identify occultation periods only once (assuming they are the same for all frequencies)
    occultation_changes = data[col].astype(int).diff()  # Detect changes in isVis
    start_times = data.index[(occultation_changes == -1)]  # False (occulted) → True (visible)
    end_times = data.index[(occultation_changes == 1)]  # True (visible) → False (occulted)

    # Ensure start_times and end_times align properly
    if len(end_times) > 0 and (len(start_times) == 0 or start_times[0] > end_times[0]):
        end_times = end_times[1:]  # Remove first end_time if it comes before any start_time
    
    if len(start_times) > len(end_times):
        start_times = start_times[:-1]  # Remove extra start_time if unmatched

    valid_pairs = [(s, e) for s, e in zip(start_times, end_times) if (e - s) >= pd.Timedelta(minutes=2)]
    start_times, end_times = zip(*valid_pairs) if valid_pairs else ([], [])

    # Loop over frequency bands
    for freq in tqdm(freqs):
        freq_data = data[data['frequency_band'] == freq]
        occultation_stats = {'median': [], 'std': []}
        non_occultation_stats = {'median': [], 'std': []}

        for start, end in zip(start_times, end_times):
            # Select occulted region
            occultation_period = freq_data.loc[(freq_data.index >= start) & (freq_data.index <= end)]

            # Select non-occulted regions before and after
            pre_occult = freq_data.loc[(freq_data.index >= start - window) & 
                                       (freq_data.index < start) & 
                                       (freq_data[col] == True)]
            
            post_occult = freq_data.loc[(freq_data.index > end) & 
                                        (freq_data.index <= end + window) & 
                                        (freq_data[col] == True)]
            
            non_occult = pd.concat([pre_occult, post_occult])  # Combine both non-occultation regions
            
            if occultation_period.empty:
                print(f"Skipping empty occultation period: {start} to {end}")
                continue

            # Extract signal values
            occultation_period_sig = sigmaClip(occultation_period[antenn],n=n)
            non_occult_sig = sigmaClip(non_occult[antenn],n=n)  
            
            if np.isnan(occultation_period_sig).any():
                print('NaN detected in occultation period')
                print(len(occultation_period[antenn]))
                print(start)
                print(end)
                continue
            if np.isnan(non_occult_sig).any():
                print('NaN detected in non-occulted period')
                print(len(non_occult[antenn]))
                print(start)
                print(end)
                continue

            # Compute statistics
            if len(occultation_period_sig) > 0:
                occultation_stats['median'].append(np.nanmedian(occultation_period_sig))
                occultation_stats['std'].append(np.nanstd(occultation_period_sig))

            if len(non_occult_sig) > 0:
                non_occultation_stats['median'].append(np.nanmedian(non_occult_sig))
                non_occultation_stats['std'].append(np.nanstd(non_occult_sig))
        
        # Store statistics for this frequency band
        stats[freq] = {
            'occulted': occultation_stats,
            'non_occulted': non_occultation_stats
        }

    return stats        


# In[16]:


def occultationHistogramPlotter(histograms):
    num_bins = len(histograms)
    
    # Calculate grid size (3x3)
    num_rows = (num_bins + 2) // 3  # Ceiling division to determine rows
    num_cols = 3
    
    # Create a figure and axes
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    axes = axes.flatten()  # Flatten the 2D grid of axes for easy iteration
    
    # Loop through each frequency bin and plot
    for i, (freq, data) in enumerate(histograms.items()):
        ax = axes[i]
        
        # Extract histogram data
        occulted_hist = data['occulted']['hist']/np.max(data['occulted']['hist'])
        occulted_bins = data['occulted']['bins']
        nonOcculted_hist = data['nonOcculted']['hist']/np.max(data['nonOcculted']['hist'])
        nonOcculted_bins = data['nonOcculted']['bins']
        
        # Plot occulted histogram
        ax.hist(
            occulted_bins[:-1], bins=occulted_bins, weights=occulted_hist, 
            alpha=0.6, label='Occulted', color='blue'
        )
        
        # Plot non-occulted histogram
        ax.hist(
            nonOcculted_bins[:-1], bins=nonOcculted_bins, weights=nonOcculted_hist, 
            alpha=0.6, label='Non-Occulted', color='orange'
        )
        
        # Set titles and labels
        ax.set_title(f'Frequency Bin: {freq}')
        ax.set_xlabel('rv1_coarse')
        ax.set_ylabel('Counts')
        ax.legend()
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


# In[17]:


def comparePower(occulted,notOcculted):
    freqs = occulted['frequency_band'].unique()
    delta = {}
    N = {}
    for freq in freqs:
        occult_freq = occulted[occulted['frequency_band'] == freq]
        notOccult_freq = notOcculted[notOcculted['frequency_band']==freq]
        if(np.average(occult_freq['rv1_coarse'])<np.average(notOccult_freq['rv1_coarse'])):
            delta[freq] = 1
        else:
            delta[freq] = -1
        N[freq] = 1
    return delta,N


# In[18]:


def plotNormalizedOccultationHistograms(stats, use_std_weights=False, min_bin_percentage=0.05, apply_filter=False, fig_label=None, save_path=None):
    freqs = stats.keys()
    num_freqs = len(freqs)

    # Define grid size (3x3 or adjust if needed)
    cols = 3
    rows = (num_freqs // cols) + (num_freqs % cols > 0)

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()  # Flatten in case we have fewer than 9 frequencies

    for i, freq in enumerate(freqs):
        ax = axes[i]

        # Extract medians and stds for the frequency
        occulted_medians = np.array(stats[freq]['occulted']['median'])
        non_occulted_medians = np.array(stats[freq]['non_occulted']['median'])

        occulted_stds = np.array(stats[freq]['occulted']['std'])
        non_occulted_stds = np.array(stats[freq]['non_occulted']['std'])

        # Remove top x% of the data if apply_filter is True
        if apply_filter:
            occ_threshold = np.percentile(occulted_medians, 100 * (1 - min_bin_percentage))
            non_occ_threshold = np.percentile(non_occulted_medians, 100 * (1 - min_bin_percentage))

            mask_occ = occulted_medians <= occ_threshold
            mask_non_occ = non_occulted_medians <= non_occ_threshold

            occulted_medians = occulted_medians[mask_occ]
            non_occulted_medians = non_occulted_medians[mask_non_occ]

            if use_std_weights:
                occulted_stds = occulted_stds[mask_occ]
                non_occulted_stds = non_occulted_stds[mask_non_occ]

        # Compute weights (inverse of std, to give higher weight to more precise measurements)
        if use_std_weights:
            occulted_weights = 1 / (occulted_stds + 1e-6)  # Add small number to avoid division by zero
            non_occulted_weights = 1 / (non_occulted_stds + 1e-6)
        else:
            occulted_weights = None
            non_occulted_weights = None

        # Use shared binning for both datasets
        combined_data = np.concatenate((occulted_medians, non_occulted_medians))
        bins = np.histogram_bin_edges(combined_data, bins=40)

        occ_counts, _ = np.histogram(occulted_medians, bins=bins, weights=occulted_weights, density=True)
        non_occ_counts, _ = np.histogram(non_occulted_medians, bins=bins, weights=non_occulted_weights, density=True)

        occ_bins_start = bins[:-1]
        occ_bins_end = bins[1:]
        non_occ_bins_start = bins[:-1]
        non_occ_bins_end = bins[1:]

        # Normalize counts so max count is 1
        if np.max(occ_counts) > 0:
            occ_counts = occ_counts / np.max(occ_counts)
        if np.max(non_occ_counts) > 0:
            non_occ_counts = non_occ_counts / np.max(non_occ_counts)

        # Plot normalized histograms
        ax.bar(occ_bins_start, occ_counts, width=occ_bins_end - occ_bins_start, alpha=0.6, label='Occulted', color='blue', align='edge')
        ax.bar(non_occ_bins_start, non_occ_counts, width=non_occ_bins_end - non_occ_bins_start, alpha=0.6, label='Non-Occulted', color='orange', align='edge')

        # Labels
        ax.set_title(f"Frequency {bandToFreq(freq)} MHz")
        ax.set_xlabel("Median Signal Strength")
        ax.set_ylabel("Normalized Count")
        ax.legend()

    # Remove unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Add figure label if provided
    if fig_label:
        fig.text(0.001, 0.98, fig_label, fontsize=20, verticalalignment='top', horizontalalignment='left')

    plt.tight_layout()

    # Save the figure if save_path is provided
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        print(f"Figure saved as {save_path}")

    plt.show()

def plotNormalizedOccultationHistogramsPretty(
        stats,
        *,
        use_std_weights=False,
        min_bin_percentage=None,
        apply_filter=False,
        share_y=False,
        palette=("tab:blue", "tab:orange"),
        suptitle=None,
        save_path=None,
):
    freqs = sorted(stats.keys())
    num_freqs = len(freqs)

    if isinstance(min_bin_percentage, (int, float)) or min_bin_percentage is None:
        min_bin_percentage = [min_bin_percentage] * num_freqs
    elif len(min_bin_percentage) != num_freqs:
        raise ValueError("min_bin_percentage must have one entry per frequency")

    cols, rows = 3, (num_freqs + 2) // 3
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4.5 * rows),
                             sharey=share_y, constrained_layout=True)
    axes = axes.flatten()

    for i, (freq, pct) in enumerate(zip(freqs, min_bin_percentage)):
        ax = axes[i]

        occ_med  = np.asarray(stats[freq]["occulted"]["median"])
        nocc_med = np.asarray(stats[freq]["non_occulted"]["median"])
        occ_std  = np.asarray(stats[freq]["occulted"]["std"])
        nocc_std = np.asarray(stats[freq]["non_occulted"]["std"])

        if apply_filter and pct is not None:
            thr_occ  = np.percentile(occ_med,  100 * (1 - pct))
            thr_nocc = np.percentile(nocc_med, 100 * (1 - pct))
            keep_occ  = occ_med  <= thr_occ
            keep_nocc = nocc_med <= thr_nocc
            occ_med,  occ_std  = occ_med[keep_occ],  occ_std[keep_occ]
            nocc_med, nocc_std = nocc_med[keep_nocc], nocc_std[keep_nocc]

        occ_w  = 1 / (occ_std  + 1e-9) if use_std_weights else None
        nocc_w = 1 / (nocc_std + 1e-9) if use_std_weights else None

        bins = np.histogram_bin_edges(np.concatenate([occ_med, nocc_med]), bins=40)
        occ_cnt, _  = np.histogram(occ_med,  bins=bins, weights=occ_w,  density=True)
        nocc_cnt, _ = np.histogram(nocc_med, bins=bins, weights=nocc_w, density=True)
        occ_cnt  /= occ_cnt.max()  if occ_cnt.max()  > 0 else 1
        nocc_cnt /= nocc_cnt.max() if nocc_cnt.max() > 0 else 1

        width = np.diff(bins)
        ax.bar(bins[:-1], occ_cnt,  width=width, align="edge",
               alpha=0.65, color=palette[0], edgecolor="black", label="Occulted")
        ax.bar(bins[:-1], nocc_cnt, width=width, align="edge",
               alpha=0.65, color=palette[1], edgecolor="black", label="Non‑Occulted")

        # ---------- cosmetics ------------------------------------
        ax.set_title(f"{bandToFreq(freq)} MHz")
        ax.set_xlabel("Median signal")
        if not share_y or i % cols == 0:
            ax.set_ylabel("Normalized count")
        ax.set_yticks(np.linspace(0, 1, 6))

        # --- force scientific notation on x‑axis, hide offset ----
        ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0), useMathText=True)
        ax.xaxis.get_offset_text().set_visible(False)

        ax.legend(frameon=False, fontsize=9)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    if suptitle:
        fig.suptitle(suptitle, fontsize=16, y=1.02)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved → {save_path}")

    plt.show()
def plotOccultationHistogramSingle(
        stats_for_freq,
        *,
        freq_key,
        use_std_weights=False,
        min_bin_percentage=None,
        apply_filter=False,
        palette=("tab:blue", "tab:orange"),
        title=None,
        save_path=None,
):
    """
    Parameters
    ----------
    stats_for_freq : dict
        stats[freq_key] slice – must contain
        ["occulted"]["median"|"std"] and ["non_occulted"][...]
    freq_key : any
        Used only for the title (e.g. 8 or "8")
    use_std_weights : bool, optional
        Weight by 1/σ if True.
    min_bin_percentage : float or None, optional
        If e.g. 0.02, drop the top 2 % of values (largest medians).
    apply_filter : bool, optional
        Whether to apply the threshold cut.
    palette : tuple(str,str), optional
        Colors for (occulted, non‑occulted).
    title : str or None, optional
        Override panel title.  Default is "freq_key MHz".
    save_path : str or Path or None
        If given, writes figure to disk.
    """

    # ---------- unpack data ---------------------------------------------------
    occ_med  = np.asarray(stats_for_freq["occulted"]["median"])
    nocc_med = np.asarray(stats_for_freq["non_occulted"]["median"])
    occ_std  = np.asarray(stats_for_freq["occulted"]["std"])
    nocc_std = np.asarray(stats_for_freq["non_occulted"]["std"])

    # ---------- optional tail trimming ---------------------------------------
    if apply_filter and min_bin_percentage is not None:
        thr_occ  = np.percentile(occ_med,  100 * (1 - min_bin_percentage))
        thr_nocc = np.percentile(nocc_med, 100 * (1 - min_bin_percentage))
        occ_mask  = occ_med  <= thr_occ
        nocc_mask = nocc_med <= thr_nocc
        occ_med,  occ_std  = occ_med[occ_mask],  occ_std[occ_mask]
        nocc_med, nocc_std = nocc_med[nocc_mask], nocc_std[nocc_mask]

    # ---------- weights -------------------------------------------------------
    occ_w  = 1 / (occ_std  + 1e-9) if use_std_weights else None
    nocc_w = 1 / (nocc_std + 1e-9) if use_std_weights else None

    # ---------- shared bins, normalized counts -------------------------------
    bins = np.histogram_bin_edges(np.concatenate([occ_med, nocc_med]), bins=40)
    occ_cnt, _  = np.histogram(occ_med,  bins=bins, weights=occ_w,  density=True)
    nocc_cnt, _ = np.histogram(nocc_med, bins=bins, weights=nocc_w, density=True)
    occ_cnt  /= occ_cnt.max()  if occ_cnt.max()  > 0 else 1
    nocc_cnt /= nocc_cnt.max() if nocc_cnt.max() > 0 else 1

    # ---------- plotting ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    width = np.diff(bins)
    ax.bar(bins[:-1], occ_cnt,  width=width, align="edge",
           alpha=0.65, color=palette[0], edgecolor="black", label="Occulted")
    ax.bar(bins[:-1], nocc_cnt, width=width, align="edge",
           alpha=0.65, color=palette[1], edgecolor="black", label="Non‑Occulted")

    ax.set_xlabel("Median signal")
    ax.set_ylabel("Normalized count")
    ax.set_title(title or f"{freq_key} MHz")
    ax.set_yticks(np.linspace(0, 1, 6))

    # scientific notation on x‑axis, no offset text
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0), useMathText=True)
    ax.xaxis.get_offset_text().set_visible(False)

    ax.legend(frameon=False)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved → {save_path}")

    plt.show()
