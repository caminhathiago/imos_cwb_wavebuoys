import math
from datetime import datetime, timedelta

import polars as pl
import dask
import numpy as np
from numpy.exceptions import ComplexWarning
import warnings

warnings.filterwarnings("ignore", category=ComplexWarning)


class Spectra:

    def __init__(self):
        self.spectra_params = ['TIME', 'FREQUENCY', 'A1', 'B1', 'A2', 'B2', 'ENERGY']
        self.bulk_params = ['TIME', 'WSSH', 'WPFM', 'WPPE', 'SSWMD', 'WPDI', 'WMDS', 'WPDS']

    def calculate_nfft(self, fs: float, segments: int = 8, spec_window: int = 30) -> float:
        return (2** int(np.ceil(np.log2(spec_window * 60 * fs / segments))))

    def calculate_min_samples(self, fs: float, spec_window: int = 30) -> float:
        return spec_window * 60 * fs

    def define_merging(self, nfft: int) -> int:
        if nfft == 512:
            return 3
        elif nfft == 1024:
            return 5
        elif nfft == 2048:
            return 7

    def ZeroUpX3(self, eta, dt):
        """
        Calculate significant wave height Hs and mean period Tz based on zero
        upcrossing of the wave record eta. Also obtain individual wave heights 
        and periods, crests and troughs.
        
        Parameters:
        eta (np.array): Array of wave heights (time series of surface elevations).
        dt (float): Time step between data points.
        
        Returns:
        dict: Dictionary containing the calculated wave properties:
            - 'Periods': List of individual wave periods.
            - 'Heights': List of individual wave heights.
            - 'Crests': List of individual wave crests.
            - 'Troughs': List of individual wave troughs.
            - 'Hs': Significant wave height.
            - 'Tz': Mean zero-upcrossing wave period.
        """

        k = np.arange(len(eta) - 1)
        p = np.where((np.sign(eta[k]) <= 0) & (np.sign(eta[k + 1]) > 0))[0]

        n = len(p) - 1
        zup = {
            'Periods': np.zeros(n),
            'Heights': np.zeros(n),
            'Crests': np.zeros(n),
            'Troughs': np.zeros(n)
        }

        for m in range(n):
            # Linear interpolation to get zero-upcrossing times
            ts = dt * (p[m] - eta[p[m]] / (eta[p[m] + 1] - eta[p[m]]))
            te = dt * (p[m + 1] - eta[p[m + 1]] / (eta[p[m + 1] + 1] - eta[p[m + 1]]))
            zup['Periods'][m] = te - ts  # wave period
            
            maxpos = np.max(eta[p[m] + 1: p[m + 1]])
            maxneg = np.min(eta[p[m] + 1: p[m + 1]])
            zup['Heights'][m] = maxpos - maxneg  # wave height
            
            zup['Crests'][m] = maxpos  # crest
            zup['Troughs'][m] = maxneg  # trough
        
        # Significant wave height (Hs)
        SortedH = np.sort(zup['Heights'])[::-1]
        zup['Hs'] = np.mean(SortedH[:round(len(zup['Heights']) / 3)])  # Average of the largest 1/3 heights
        
        # Mean zero-upcrossing wave period (Tz)
        zup['Tz'] = np.mean(zup['Periods'])
        
        return zup

    def spectra_from_displacements(self, heave, north, east, nfft, nover, fs, merge, data_type, info):
        # FIRST SPLIT DATA INTO CHUNKS
        pts = len(heave)  # record length in data points
        windows = math.floor((1 / nover) * (pts / nfft - 1) + 1)  # number of windows/segments

        # COMPUTE ZERO UP CROSSING WAVE HEIGHTS FOR OUTLIER DETECTION
        zup = self.ZeroUpX3(heave, 1 / fs)  # NOTE - use complete record

        Hs0 = zup["Hs"]
        heights = zup["Heights"]
        periods = zup["Periods"]
        T0 = zup["Tz"]
        
        # Initialize arrays to store the segments
        hv_segs = np.zeros((nfft, windows))  # Heave segments
        nt_segs = np.zeros((nfft, windows))  # North segments
        et_segs = np.zeros((nfft, windows))  # East segments
        
        # Initialize lists to store results for each window
        Hs_seg = np.zeros(windows)  # Significant wave heights for each segment
        heights_seg = []  # Heights for each segment
        periods_seg = []  # Periods for each segment
        T0_seg = np.zeros(windows)  # Wave periods for each segment

        for q in range(windows):
            start_idx = int((q) * (nover * nfft))  # Start index for the segment
            end_idx = start_idx + nfft  # End index for the segment
            
            # Extract data segments
            hv_segs[:, q] = heave[start_idx:end_idx]
            nt_segs[:, q] = north[start_idx:end_idx]
            et_segs[:, q] = east[start_idx:end_idx]
            
            # Zero crossing analysis for each segment
            zup_seg = self.ZeroUpX3(hv_segs[:, q], 1 / fs)
            
            # Store results for the segment
            Hs_seg[q] = zup_seg["Hs"]
            heights_seg.append(zup_seg["Heights"])
            periods_seg.append(zup_seg["Periods"])
            T0_seg[q] = zup_seg["Tz"]

        rw = [
        window for window in range(windows)
        if np.any(np.isnan(np.concatenate([hv_segs[:, window], nt_segs[:, window], et_segs[:, window]]))) or
            max(heights_seg[window]) > info["hs0_thresh"] * Hs0 or
            max(periods_seg[window]) > info["t0_thresh"] * T0 or
            max(periods_seg[window]) > 30
    ]

        if len(rw) == 0 or (len(rw) < windows * (1 - info['bad_data_thresh']) and np.sum(heave == 0) / len(heave) < 0.1):
            
            if len(rw) > 0:
                # Remove bad segments (those with bad data)
                hv_segs = np.delete(hv_segs, rw, axis=1)
                nt_segs = np.delete(nt_segs, rw, axis=1)
                et_segs = np.delete(et_segs, rw, axis=1)

        # MAKE WINDOW
        win = np.hanning(nfft)
        win = win.reshape((nfft, 1))

        # APPLY WINDOW
        hv_win = win * hv_segs  # Element-wise multiplication
        nt_win = win * nt_segs  # Element-wise multiplication
        et_win = win * et_segs

        # Compute correction factors based on variance preservation
        facth = np.sqrt(np.var(hv_segs, axis=0) / np.var(hv_win, axis=0))
        factn = np.sqrt(np.var(nt_segs, axis=0) / np.var(nt_win, axis=0))
        facte = np.sqrt(np.var(et_segs, axis=0) / np.var(et_win, axis=0))

        hv_corr = (np.ones((nfft, 1)) * facth) * hv_win
        nt_corr = (np.ones((nfft, 1)) * factn) * nt_win
        et_corr = (np.ones((nfft, 1)) * facte) * et_win

        # Compute 2-sided FFT
        hfft = np.fft.fft(hv_corr, axis=0)  # FFT for heave data
        nnfft = np.fft.fft(nt_corr, axis=0)  # FFT for north data
        efft = np.fft.fft(et_corr, axis=0)  # FFT for east data

        # Delete second half of the spectrum
        hfft = hfft[:nfft//2, :]
        nnfft = nnfft[:nfft//2, :]
        efft = efft[:nfft//2, :]

        # Assuming hfft, nnfft, efft are numpy arrays with shape (512, 7)
        # First, remove the first row (index 0) from each matrix
        hfft = hfft[1:, :]
        nnfft = nnfft[1:, :]
        efft = efft[1:, :]

        # Now, append a small value to the end of each matrix to avoid NaNs
        small_value = 1e-10
        hfft = np.vstack([hfft, np.zeros((1, hfft.shape[1])) + small_value])
        nnfft = np.vstack([nnfft, np.zeros((1, nnfft.shape[1])) + small_value])
        efft = np.vstack([efft, np.zeros((1, efft.shape[1])) + small_value])

        # POWER SPECTRA (auto-spectra)
        hh_spec = hfft * np.conj(hfft)  # Heave auto-spectra
        nn_spec = nnfft * np.conj(nnfft)  # North auto-spectra
        ee_spec = efft * np.conj(efft) # East auto-spectra

        # CROSS-SPECTRA
        he_spec = hfft * np.conj(efft)  # Heave-East cross-spectra
        hn_spec = hfft * np.conj(nnfft)  # Heave-North cross-spectra
        en_spec = efft * np.conj(nnfft) # East-North cross-spectra

        # Merge frequency bands, set merge=1 for no merging
        if merge > 1:
            num_bands = nfft // (2 * merge)  # Calculate number of merged bands
            
            hh_spec_merged = np.zeros((num_bands, hh_spec.shape[1])) + 1e-10
            nn_spec_merged = np.zeros((num_bands, nn_spec.shape[1])) + 1e-10
            ee_spec_merged = np.zeros((num_bands, ee_spec.shape[1])) + 1e-10

            he_spec_merged = np.zeros((num_bands, he_spec.shape[1]), dtype=complex) + 1e-10
            hn_spec_merged = np.zeros((num_bands, hn_spec.shape[1]), dtype=complex) + 1e-10
            en_spec_merged = np.zeros((num_bands, en_spec.shape[1]), dtype=complex) + 1e-10

            # Loop through frequency bands, ensuring no out-of-bounds indexing
            for i in range(merge, nfft // 2, merge):
                band_idx = i // merge  # Index for the merged spectrum band
                
                
                if band_idx <= num_bands:
                    # Auto-spectra
                    hh_spec_merged[band_idx-1, :] = np.mean(hh_spec[i - merge:i, :], axis=0)
                    nn_spec_merged[band_idx-1, :] = np.mean(nn_spec[i - merge:i, :], axis=0)
                    ee_spec_merged[band_idx-1, :] = np.mean(ee_spec[i - merge:i, :], axis=0)

                    # Cross-spectra
                    he_spec_merged[band_idx-1, :] = np.mean(he_spec[i - merge:i, :], axis=0)
                    hn_spec_merged[band_idx-1, :] = np.mean(hn_spec[i - merge:i, :], axis=0)
                    en_spec_merged[band_idx-1, :] = np.mean(en_spec[i - merge:i, :], axis=0)

        else:
            hh_spec_merged = hh_spec
            nn_spec_merged = nn_spec
            ee_spec_merged = ee_spec

            he_spec_merged = he_spec
            hn_spec_merged = hn_spec
            en_spec_merged = en_spec

        # Frequency range and bandwidth
        n = (nfft / 2) / merge  # Number of frequency bands
        Nyquist = 0.5 * fs  # Highest spectral frequency
        bandwidth = Nyquist / n  # Frequency (Hz) bandwidth

        # Find the middle of each frequency band, only works when merging an odd number of bands!
        freq = 1 / nfft + bandwidth / 2 + bandwidth * np.arange(int(n))  # Middle of each frequency band

        # Ensemble Average
        s = 2 * np.mean(hh_spec_merged, axis=1) / (nfft * fs)
        uu = 2 * np.mean(ee_spec_merged, axis=1) / (nfft * fs)
        vv = 2 * np.mean(nn_spec_merged, axis=1) / (nfft * fs)

        hu = 2 * np.mean(he_spec_merged, axis=1) / (nfft * fs)
        hv = 2 * np.mean(hn_spec_merged, axis=1) / (nfft * fs)
        uv = 2 * np.mean(en_spec_merged, axis=1) / (nfft * fs)
        
        # Calculate Co-Spectra (real part)
        coHU = np.real(hu)
        coHV = np.real(hv)
        coUV = np.real(uv)

        # Calculate Quad-Spectra (imaginary part)
        qHU = np.imag(hu)
        qHV = np.imag(hv)
        qUV = np.imag(uv)

        # Spectral moments - a1, a2, b1, b2
        if data_type == 'xyz':  # xyz = displacements
            a1 = qHU / np.sqrt(s * (uu + vv))
            b1 = qHV / np.sqrt(s * (uu + vv))
        elif data_type == 'enu':  # enu = velocity
            a1 = coHU / np.sqrt(s * (uu + vv))
            b1 = coHV / np.sqrt(s * (uu + vv))

        # Calculate a2 and b2
        a2 = (uu - vv) / (uu + vv)
        b2 = (2 * coUV) / (uu + vv)
        
        # Calculate check factor - ratio of horizontal to vertical displacements in each frequency bin
        # For linear waves in deep water should be 1.
        # NOTE: often seen written in the inverse with vertical in numerator,
        # but this has the disadvantage of blowing up for frequencies with near 0 energy
        Check = (uu + vv) / s

        # Ensure a1 and b1 are numeric (float) arrays
        a1 = np.asarray(a1, dtype=np.float64)
        b1 = np.asarray(b1, dtype=np.float64)
        
        a2 = np.asarray(a2, dtype=np.float64)
        b2 = np.asarray(b2, dtype=np.float64)
        
        # Primary directional spectrum --- direction at each frequency
        dir1 = np.rad2deg(np.arctan2(b1, a1))
        spread1 = np.sqrt(2 * (1 - np.sqrt(a1**2 + b1**2)))

        # Secondary directional spectrum --- direction at each frequency
        dir2 = np.rad2deg(np.arctan2(b2, a2) / 2)
        spread2 = np.sqrt(np.abs(0.5 - 0.5 * (a2 * np.cos(2 * np.deg2rad(dir2)) + b2 * np.sin(2 * np.deg2rad(dir2)))))
        
        # Mean directions - total, SS, IG
        mdir1_tot = np.rad2deg(np.arctan2(np.nansum((s * b1).real),np.nansum((s * a1).real)))
        mdir2_tot = np.rad2deg(np.arctan2(np.nansum((s * b2).real), np.nansum((s * a2).real)) / 2)

        # Rotate in WAVES FROM
        mdir1_tot = np.mod(270 - mdir1_tot, 360)
        mdir2_tot = np.mod(270 - mdir2_tot, 360)
        
        # Peak period and direction
        ff = np.argmax(s)  # Find index where S is maximum
        Tp = 1.0 / freq[ff]
        Dp = np.rad2deg(np.arctan2(b1[ff], a1[ff]))
        Dp = np.mod(270 - Dp, 360)
        spread_Dp = np.rad2deg(np.sqrt(2 * (1 - np.sqrt(a1[ff]**2 + b1[ff]**2))))

        # Method following Rogers and Wang eq. 7
        a1_bar = np.trapezoid(a1 * s, freq) / np.trapezoid(s, freq)
        b1_bar = np.trapezoid(b1 * s, freq) / np.trapezoid(s, freq)
        spread = (180 / np.pi) * np.sqrt(2 * (1 - np.sqrt(a1_bar**2 + b1_bar**2)))

        # return (pts, windows, zup,
        #         hv_segs, win, hv_win,
        #         facth, hv_corr,
        #         hfft, hh_spec, hh_spec_merged,
        #         he_spec, he_spec_merged,
        #         freq, a1, b1, a2, b2, 
        #         s, uu, vv, hu, hv, uv, 
        #         coHU, coUV, qHU, qHV, qUV, 
        #         dir1, spread1, dir2, spread2, 
        #         mdir1_tot, mdir2_tot,
        #         ff, freq[ff],
        #         Tp, Dp, spread_Dp, 
        #         a1_bar, b1_bar, spread
        #         )
        # Calculate wave parameters and organize output
        out = {}
        spectra = {}
        if data_type == 'xyz':  # xyz = displacements
            # Calculate moments of spectrum - total
            n = np.array([0, 1, 2, 3])
            m = np.zeros(4)
            for jj in range(4):
                # Calculate moments
                m[jj] = np.trapezoid(freq**n[jj] * s, freq)
                if n[jj] == 0:
                    out['Hm0'] = 4 * np.sqrt(m[jj])  # Significant wave height
            
            SpTot = np.trapezoid(s, freq)
            out['Hrms'] = np.sqrt(8 * SpTot)
            spectra['f'] = freq
            spectra['spec1D'] = s
            spectra['a1'] = a1
            spectra['a2'] = a2
            spectra['b1'] = b1
            spectra['b2'] = b2
            out['mdir1'] = mdir1_tot
            out['mdir2'] = mdir2_tot
            out['mdir1_spec'] = np.mod(270 - dir1, 360)
            out['Tp'] = Tp
            out['Tm1'] = m[0] / m[1]  # m0/m1
            out['Tm2'] = np.sqrt(m[0] / m[2])  # m0/m2
            out['Dp'] = Dp
            out['spread_Dp'] = spread_Dp
            out['spread'] = spread
            out['spread_spec'] = spread1
            out['Check'] = Check
            out['segments'] = windows
            out['segments_used'] = windows - len(rw)
            out['zup'] = zup
        # elif type == 'enu':  # enu = velocity
        #     # Apply depth correction
        #     depth = info['hab'] + np.mean(heave)
        #     wnum = disperk(freq, depth)
        #     # Transformation to surface elevation variance spectrum
        #     scorr = s * (np.cosh(wnum * depth) / np.cosh(wnum * info['hab']))**2
            
        #     # Calculate moments of spectrum
        #     m = np.zeros(4)
        #     for jj in range(4):
        #         # Calculate moments
        #         m[jj] = np.trapezoid(freq**n[jj] * scorr, freq)
        #         if n[jj] == 0:
        #             out['Hm0'] = 4 * np.sqrt(m[jj])  # Significant wave height
            
        #     SpTot = np.trapz(scorr, freq)
        #     out['f'] = freq
        #     out['spec1D'] = scorr
        #     out['a1'] = a1
        #     out['a2'] = a2
        #     out['b1'] = b1
        #     out['b2'] = b2
        #     out['mdir1_spec'] = np.mod(270 - dir1, 360)
        #     out['mdir1'] = mdir1_tot
        #     out['mdir2'] = mdir2_tot
        #     out['Tp'] = Tp
        #     out['Tm1'] = m[0] / m[1]  # m0/m1
        #     out['Tm2'] = np.sqrt(m[0] / m[2])  # m0/m2
        #     out['Dp'] = Dp
        #     out['spread'] = spread
        #     out['spread_spec'] = spread1
        #     out['Check'] = Check
        #     out['segments'] = windows
        #     out['segments_used'] = windows - len(rw)

        else:  # Too much missing/bad data, do not compute spectrum
            out, spectra = np.nan, np.nan

        return out, spectra

    def generate_time_chunks(self, data:pl.DataFrame, time_chunk:str = '30m') -> pl.DataFrame:
        return (data
                      .group_by_dynamic("datetime", every=time_chunk, closed="left")
                      .agg([pl.all(), pl.col("datetime").first().alias("chunk_datetime")])
            )
    
    def filter_insufficient_samples(self, data:pl.DataFrame, min_samples:float) -> pl.DataFrame:
        return data.filter(pl.col("z").list.len() >= min_samples)
         
    def generate_dask_chunks(self, data: pl.DataFrame, num_workers: int):
        """
        Splits a Polars dataframe into chunks, adding any remainder rows to the last chunk.
        
        Args:
            dataframe (pl.DataFrame): The Polars dataframe to split.
            num_workers (int): The number of chunks to create.
        
        Returns:
            list of pl.DataFrame: A list of Polars dataframe chunks.
        """
        total_rows = len(data)
        chunk_size = total_rows // num_workers

        dask_chunks = []
        start_idx = 0

        for i in range(num_workers):
            if i == num_workers - 1: 
                end_idx = total_rows
            else:
                end_idx = start_idx + chunk_size
            dask_chunks.append(data[start_idx:end_idx])
            start_idx = end_idx

        return dask_chunks

    def process_dask_chunk(self, dask_chunk, nfft, nover, fs, merge, data_type, info):
        start_time = dask_chunk.select(pl.col("datetime").min()).item()
        end_time = dask_chunk.select(pl.col("datetime").max()).item() 
        
        results = {"TIME": [], "FREQUENCY": [],
                   "A1": [], "B1": [], "A2": [], "B2": [], "ENERGY": [],
                   'WSSH':[], 'WPFM':[], 'WPPE':[], 'SSWMD':[], 'WPDI':[], 'WMDS':[], 'WPDS':[]}

        chunk_size = timedelta(minutes=30)

        current_time = start_time

        while current_time <= end_time:
            data_chunk = dask_chunk.filter(
                    (pl.col("datetime") == current_time)
                )
            
            if data_chunk["z"].is_empty():
                current_time += timedelta(minutes=30)
                continue
            
            out, spectra = self.spectra_from_displacements(data_chunk["z"].item().to_numpy(),
                                            data_chunk["y"].item().to_numpy(), 
                                            data_chunk["x"].item().to_numpy(),
                                            nfft, nover, fs, merge, data_type, info)
            
            results["TIME"].append(current_time)
            results["FREQUENCY"].append(spectra["f"])
            results["A1"].append(spectra["a1"])
            results["B1"].append(spectra["b1"])
            results["A2"].append(spectra["a2"])
            results["B2"].append(spectra["b2"])
            results["ENERGY"].append(spectra["spec1D"])

            results["WSSH"].append(out["Hm0"])
            results["WPFM"].append(out["Tm1"])
            results["WPPE"].append(out["Tp"])
            results["SSWMD"].append(out["mdir1"])
            results["WPDI"].append(out["Dp"])
            results["WMDS"].append(out["spread"])
            results["WPDS"].append(out["spread_Dp"])
            
            current_time += chunk_size

        return pl.DataFrame(results)

    def select_parameters(self, dataframe:pl.DataFrame, dataset_type:str = "spectra") -> pl.DataFrame:
        if dataset_type == "spectra":
            cols = self.spectra_params
        if dataset_type == "bulk":
            cols = self.bulk_params
        
        return dataframe.select(cols)