function X = myFFT(x)
    x = x(:);              % force column vector
    N = length(x);
    X = zeros(N,1);

    for k = 0:N-1
        sum_val = 0;
        for n = 0:N-1
            sum_val = sum_val + x(n+1) * exp(-1j*2*pi*k*n/N);
        end
        X(k+1) = sum_val;
    end
end
