function X = myFFT_2(x, N_fft)
    x = x(:);              % force column vector
    N_input = length(x);
    
    % If FFT length not specified, use input length
    if nargin < 2
        N_fft = N_input;
    end
    
    % Zero-padding or truncation
    if N_fft > N_input
        % Zero-pad if FFT length is larger
        x = [x; zeros(N_fft - N_input, 1)];
    elseif N_fft <= N_input
        % Truncate if FFT length is smaller
        x = x(1:N_fft);
    end
    
    % Perform FFT with length N_fft
    X = zeros(N_fft, 1);
    
    for k = 0:N_fft-1
        sum_val = 0;
        for n = 0:N_fft-1
            sum_val = sum_val + x(n+1) * exp(-1j*2*pi*k*n/N_fft);
        end
        X(k+1) = sum_val;
    end
end