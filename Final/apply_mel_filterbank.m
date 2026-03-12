function mel_energies = apply_mel_filterbank(power_spectrum, fs, nfft, num_filters)
% APPLY_MEL_FILTERBANK - Apply mel filterbank to power spectrum
% INPUTS:
%   power_spectrum - power spectrum matrix (num_bins × num_frames)
%                    Each COLUMN is one frame's power spectrum
%   fs             - sampling frequency (Hz), e.g., 16000
%   nfft           - FFT size used to compute power_spectrum
%   num_filters    - number of mel filters (typically 20-40, common: 26)
%
% OUTPUT:
%   mel_energies - mel filterbank energies (num_filters × num_frames)
%                  Each COLUMN is one frame's mel energies

[num_bins, num_frames] = size(power_spectrum);

expected_bins = nfft/2 + 1;
if num_bins ~= expected_bins
    error(['Power spectrum has %d bins but expected %d bins for nfft=%d.\n' ...
           'Make sure you used the same nfft in compute_power_spectrum().'], ...
          num_bins, expected_bins, nfft);
end

filterbank = melfb(num_filters, nfft, fs);

% Validate filterbank dimensions
[fb_rows, fb_cols] = size(filterbank);

if fb_rows ~= num_filters
    error('Filterbank has %d rows but expected %d (num_filters).', ...
          fb_rows, num_filters);
end

if fb_cols ~= num_bins
    error('Filterbank has %d columns but power_spectrum has %d bins.', ...
          fb_cols, num_bins);
end


% Apply mel filterbank via matrix multiplication
% We have:
%   filterbank:     (num_filters × num_bins)    e.g., (26 × 257)
%   power_spectrum: (num_bins × num_frames)     e.g., (257 × 131)
%
% Matrix multiply: (26 × 257) × (257 × 131) = (26 × 131)
% Result: each COLUMN is one frame's mel energies

mel_energies = filterbank * power_spectrum;
% Result: (num_filters × num_frames)

end


% TECHNICAL NOTES:
% Your technical notes are EXCELLENT - keep them all!
% 
% Just update note #4 to reflect correct dimensions:
%
% 4. Matrix multiplication explanation:
%    power_spectrum: (num_frames × num_bins)    - each ROW is one frame
%    filterbank:     (num_filters × num_bins)   - each ROW is one filter
%    
%    To compute mel energies, we multiply:
%       (num_frames × num_bins) × (num_bins × num_filters)
%       = (num_frames × num_filters)
%    
%    This requires filterbank TRANSPOSED:
%       mel_energies = power_spectrum * filterbank'
%    
%    For frame i and mel band j:
%       mel_energies[i, j] = Σ_k power_spectrum[i, k] × filterbank[j, k]
%    
%    This sums the power across all frequency bins, weighted by the
%    triangular filter, giving the total energy in that mel band.