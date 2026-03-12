function delta = mfcc_delta(mfcc, N)
    if nargin < 2, N = 2; end
    
    [T, D] = size(mfcc);
    delta = zeros(T, D);
    denom = 2 * sum((1:N).^2);
    
    for n = 1:N
        t_plus  = min(T, (1:T) + n);
        t_minus = max(1, (1:T) - n);
        delta = delta + n * (mfcc(t_plus,:) - mfcc(t_minus,:));
    end
    
    delta = delta / denom;
end