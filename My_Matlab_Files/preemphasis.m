function emphasized = preemphasis(signal, alpha)

if nargin < 2
    alpha = 0.97;  % default value (standard in speech processing)
end

% Ensure signal is a column vector
if isrow(signal)
    signal = signal(:);
end

emphasized = filter([1 -alpha], 1, signal);

end

