function log_mel = apply_log(mel_energies)

epsilon = eps;  % Use MATLAB's eps (safest, smallest)

log_mel = log(mel_energies + epsilon);

% fprintf('Logarithm applied:\n');
% fprintf('  Input size:  %d × %d\n', size(mel_energies,1), size(mel_energies,2));
% fprintf('  Epsilon:     %.2e\n', epsilon);
% fprintf('  Input range: [%.2e, %.2e]\n', min(mel_energies(:)), max(mel_energies(:)));
% fprintf('  Output range: [%.2f, %.2f]\n', min(log_mel(:)), max(log_mel(:)));
% 
% % Check for problematic values
% if any(isinf(log_mel(:)))
%     warning('%d Inf values detected! Epsilon may be too small.', ...
%             sum(isinf(log_mel(:))));
% end
% if any(isnan(log_mel(:)))
%     warning('%d NaN values detected!', sum(isnan(log_mel(:))));
% end

end

% TECHNICAL NOTES:
% 1. Why take the logarithm?
%    Human perception of loudness follows the Weber-Fechner law:
%       Perceived loudness ∝ log(Intensity)
%    
%    A sound 10× more intense doesn't sound 10× louder — it sounds like
%    a small increase. Taking the log transforms the features so that
%    equal perceptual differences correspond to equal numerical differences.
%    
%    This is the same principle behind the decibel scale:
%       dB = 10 * log₁₀(P / P_ref)
%
% 2. log() vs log10():
%    In MATLAB:
%       log(x) = natural logarithm (base e)
%       log10(x) = common logarithm (base 10)
%       log2(x) = binary logarithm (base 2)
%    
%    For MFCC, the convention is to use natural log (ln). The choice
%    doesn't fundamentally matter because:
%       log₁₀(x) = log(x) / log(10) = 0.4343 * log(x)
%    
%    It's just a global scale factor, which the DCT treats linearly.
%    However, if you're comparing your MFCC values to a reference
%    implementation, make sure you use the same base!
%
% 3. Why add epsilon?
%    If any mel_energies value is exactly zero (which can happen in
%    silent regions or due to numerical precision), then:
%       log(0) = -Inf
%    
%    This would propagate through the DCT and corrupt the entire MFCC
%    vector for that frame. Adding a tiny epsilon ensures we get a
%    very negative number instead of -Inf:
%       log(1e-16) ≈ -36.8  (large negative, but finite)
%    
%    For non-zero values, adding epsilon has negligible effect:
%       log(100 + 1e-16) ≈ log(100) = 4.605
%
% 4. Alternative: flooring
%    Instead of adding epsilon, some implementations use a floor:
%       log_mel = log(max(mel_energies, floor_value))
%    
%    This explicitly sets a minimum energy level. The two approaches
%    are functionally equivalent; epsilon is more common.
%
% 5. Dynamic range compression example:
%    Suppose we have two mel energies: 1.0 and 1000.0 (1000× ratio).
%    
%    Without log:
%       1.0 → 1.0
%       1000.0 → 1000.0
%       (Range: 1000, highly skewed)
%    
%    With log:
%       1.0 → 0.0
%       1000.0 → 6.91
%       (Range: 6.91, much more balanced)
%    
%    This compression makes the features more robust: a loud speaker
%    and a quiet speaker produce features in a similar numerical range,
%    differing by spectral shape rather than overall amplitude.
%
% 6. Relation to cepstrum:
%    The word "cepstrum" comes from reversing the first four letters of
%    "spectrum". The cepstrum is defined as:
%       cepstrum = IFFT(log(|FFT(signal)|))
%    
%    We're computing a variant called the "mel-frequency cepstrum":
%    - Instead of log(|FFT|), we use log(mel_energies)
%    - Instead of IFFT, we'll use DCT (next step)
%    
%    The log is crucial because it converts convolution in the time
%    domain into addition in the cepstral domain. This separates the
%    vocal tract filter (slow variation → low quefrency) from the
%    glottal source (fast variation → high quefrency).
%
% 7. Effect on zeros and small values:
%    In practice, mel_energies should never be exactly zero for real
%    speech (there's always some background noise). But in:
%    - Completely silent regions
%    - Edge frames with little energy
%    - Numerical precision limits
%    
%    You might get very small or zero values. The epsilon prevents
%    these from breaking your MFCC computation.
%
% 8. Why not normalize?
%    Some implementations normalize the mel energies before taking the
%    log (divide by sum or max). We don't do that here because:
%    (a) It's non-standard for MFCC
%    (b) Absolute loudness information can be useful
%    (c) Normalization can be done later if needed (cepstral mean
%        normalization is common after the DCT)

