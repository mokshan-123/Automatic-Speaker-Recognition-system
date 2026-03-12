function [FFT_mat_power,Melbank_filtered_FFT,MFC_coefficients,Final_feature_vector]=get_FFTP_Melbank_MFCC_Featurevector(frames,FFT_len,Melbank_filter)

    %Apply a window
    windowed_frames=apply_window(frames);

    %Apply FFT
    % Vectorized version - much faster!
    FFT_mat = fft(windowed_frames, FFT_len, 1);

    %Get absolute Value of the FFT
    FFT_mat_abs=abs(FFT_mat);

    %Get power spectrum
    FFT_mat_power=FFT_mat_abs(1:(FFT_len/2)+1,:).^2;

    %Apply Melbank Filter
    Melbank_filtered_FFT=Melbank_filter*FFT_mat_power;

    %Apply log 
    Log_applied_MelBank_filtered_FFT=apply_log(Melbank_filtered_FFT);

    %get MFCC
    MFC_coefficients=apply_dct(Log_applied_MelBank_filtered_FFT,13);

    %get delta MFCC
    delta_MFCC=delta(MFC_coefficients);

    %get delta_delta mfcc
    delta_delta_MFCC=delta(delta_MFCC);

    %Get feature vector
    Final_feature_vector=[MFC_coefficients,delta_MFCC,delta_delta_MFCC];
end

