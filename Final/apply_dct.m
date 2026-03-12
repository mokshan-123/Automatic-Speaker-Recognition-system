function mfcc_coeffs = apply_dct(log_mel, num_coeffs)

[~, num_filters] = size(log_mel);

% Validate num_coeffs
if num_coeffs < 1
    error('num_coeffs must be >= 1');
end

if num_coeffs > num_filters
    error('num_coeffs (%d) cannot be > num_filters (%d)', ...
          num_coeffs, num_filters);
end

dct_result = dct(log_mel);

dct_coeffs = dct_result(1:num_coeffs, :);

mfcc_coeffs = dct_coeffs;

end

% TECHNICAL NOTES:
% 1. What is the DCT?
%    The Discrete Cosine Transform is similar to the DFT, but uses only
%    cosine basis functions (no complex exponentials). For real-valued,
%    symmetric sequences, the DCT is equivalent to the real part of the DFT.
%    
%    There are 4 types of DCT (I-IV). MATLAB's dct() uses Type-II:
%       X[k] = Σ(n=0 to N-1) x[n] * cos(π*k*(2n+1) / (2N))
%    
%    This is the most common variant, used in JPEG compression, MP3
%    encoding, and (most relevant here) MFCC computation.
%
% 2. Why DCT after log-mel energies?
%    The log-mel energies are CORRELATED: neighboring mel bands have
%    similar values because they overlap in frequency. Correlated features
%    waste capacity in machine learning models.
%    
%    The DCT decorrelates the features — it finds a set of basis functions
%    (cosines at different frequencies) and expresses the log-mel energies
%    as a weighted sum of these basis functions. The weights are the
%    MFCC coefficients.
%    
%    After the DCT, most of the energy is concentrated in the first few
%    coefficients. The later coefficients capture fine detail that varies
%    more with noise than with speaker identity.
%
% 3. Energy compaction:
%    If you plot the magnitude of DCT coefficients:
%       |c[0]| > |c[1]| > |c[2]| > ... > |c[25]|  (roughly)
%    
%    The first coefficient c[0] has the largest magnitude (it's related
%    to the average log-energy). The remaining coefficients decay rapidly.
%    This is called "energy compaction" — most information is in the
%    first few coefficients.
%    
%    That's why we keep only 12-13 coefficients and discard the rest.
%
% 4. Physical meaning of coefficients:
%    - c[0]: Average log-energy (DC term, proportional to loudness)
%    - c[1]: Broad spectral tilt (low vs high frequency energy balance)
%    - c[2]: Next level of detail (captures formant-like peaks/valleys)
%    - c[3-12]: Progressively finer spectral details
%    - c[13+]: Very fine details, mostly noise
%    
%    For speaker recognition, we usually keep c[0] through c[12] (13 total).
%    Some implementations drop c[0] because it's loudness-dependent and
%    doesn't carry speaker-specific information, leaving 12 coefficients.
%
% 5. Why 13 coefficients?
%    This is empirical! Through experimentation in the 1980s-1990s,
%    researchers found that 12-13 coefficients give the best trade-off:
%    - Enough to capture vocal tract shape (formants, spectral envelope)
%    - Few enough to avoid overfitting to noise or speaking style
%    - Computationally cheap
%    
%    You can use more (15-20) or fewer (8-10), but 12-13 is standard.
%
% 6. Relation to cepstrum:
%    The full cepstrum is defined as:
%       cepstrum = IFFT(log|FFT(signal)|)
%    
%    The DCT is closely related to the DFT/FFT — it's essentially the
%    real part for symmetric sequences. So MFCC is a variant of the
%    cepstrum, computed on mel-spaced frequency bins instead of linear bins.
%    
%    The term "quefrency" (reverse of "frequency") is used for the
%    independent variable of the cepstrum. Low quefrency = slow variation
%    in the log spectrum (formants, vocal tract). High quefrency = fast
%    variation (pitch harmonics, glottal source).
%    
%    By keeping only low-quefrency coefficients (c[0-12]), we capture
%    the vocal tract shape and suppress the pitch harmonics.
%
% 7. Normalization:
%    MATLAB's dct() has an optional 'norm' parameter:
%       dct(x)          : no normalization (default)
%       dct(x, [], 'n') : no normalization (same as default, explicit)
%       dct(x, [], 'o') : orthonormal (energy-preserving)
%    
%    For MFCC, the standard is NO normalization (default behavior).
%    Some implementations use orthonormal, which only changes the scale
%    of the coefficients (not their relative values).
%
% 8. Output orientation:
%    This function returns (num_frames × num_coeffs) by default.
%    Each ROW is one frame's MFCC vector.
%    
%    Some implementations (and some MATLAB toolboxes) use the opposite:
%    (num_coeffs × num_frames), where each COLUMN is one frame.
%    
%    Check what your train.m expects! You can transpose the output
%    if needed. This is the #1 source of dimension mismatch errors.
%
% 9. Coefficient indexing: 0 vs 1
%    In the literature, MFCC coefficients are typically 0-indexed:
%       c₀, c₁, c₂, ..., c₁₂
%    
%    In MATLAB (1-indexed), they're stored as:
%       mfcc(:,1), mfcc(:,2), ..., mfcc(:,13)
%    
%    So MATLAB's mfcc(:,1) corresponds to c₀ in the literature.
%    Keep this in mind when reading papers!
%
% 10. Comparison with other transforms:
%     - DFT: Complex-valued, circular convolution, fast via FFT
%     - DCT: Real-valued, symmetric extension, used for compression
%     - DST: Real-valued, antisymmetric extension
%     - WHT: Binary-valued (Walsh-Hadamard), very fast but less optimal
%     
%     For MFCC, the DCT is preferred because:
%     (a) Real-valued (no complex arithmetic needed)
%     (b) Good energy compaction (most energy in first few coeffs)
%     (c) Fast computation (same O(N log N) as FFT)

