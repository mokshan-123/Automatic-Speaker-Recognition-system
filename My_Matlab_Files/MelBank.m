function H=MelBank(fs,FFT_length,Num_of_Mel_filters)
    Lower_f=0;
    Upper_f=fs/2;

    m_min=2595*log10(1+(Lower_f/700));
    m_max=2595*log10(1+(Upper_f/700));

    mel_array=zeros(1,Num_of_Mel_filters+2);
    for i=0:Num_of_Mel_filters+1
        mel_array(i+1)=m_min+i*((m_max-m_min)/(Num_of_Mel_filters+1));
    end

    frequency_array=700*((10.^(mel_array/2595))-1);

    k=FFT_length*frequency_array/fs;

    % Each row is one Mel filter
    % Columns = FFT bins
    H = zeros(Num_of_Mel_filters, FFT_length/2 + 1);
    for m = 1:Num_of_Mel_filters
        % k(m), k(m+1), k(m+2) are the left, center, right FFT bins of the triangle
        k_left   = floor(k(m));
        k_center = floor(k(m+1));
        k_right  = floor(k(m+2));
    
        % Rising slope (left side)
        for i = k_left:k_center
            H(m, i+1) = (i - k_left) / (k_center - k_left);
        end
    
        % Falling slope (right side)
        for i = k_center:k_right
            H(m, i+1) = (k_right - i) / (k_right - k_center);
        end
    end

end
