[data,fs]=audioread("s3.wav");

%Devide the entire track in to frames with some overlap.
frame_length = 512;
overlap=0.5;
frames=buffer(data,frame_length,round(overlap*frame_length),'nodelay');

%Apply a window
windowed_frames=apply_window(frames);

%Get the frame length and the number of frames
[frame_length_windowed,number_of_frames]=size(windowed_frames);

%Apply FFT
FFT_mat=zeros(frame_length_windowed,number_of_frames);
for i=1:number_of_frames
    FFT_mat(:,i)=myFFT(windowed_frames(:,i));
end

%Get absolute Value of the FFT
FFT_mat_abs=abs(FFT_mat);

%Get power spectrum
FFT_mat_power=FFT_mat_abs(1:(frame_length/2)+1,:).^2;

%Apply Mel Banks
Num_of_Mel_filters=20;
Melbank_filter=MelBank(fs,frame_length,Num_of_Mel_filters);

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

%get Normalized feature vector
Normalized_feature_vector=Final_feature_vector-mean(Final_feature_vector);


%get Zscore
Z_score_feature_vector=zscore(Final_feature_vector);

disp(Final_feature_vector);
disp(Normalized_feature_vector);
disp(Z_score_feature_vector);