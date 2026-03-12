function log_mel = apply_log(mel_energies)

epsilon = eps;  % Use MATLAB's eps (safest, smallest)
log_mel = log(mel_energies + epsilon);

end


