[data,fs]=audioread("s3.wav");

frame_length = 512;
overlap=0.5;
frames=buffer(data,frame_length,round(overlap*frame_length),'nodelay');

Mel_filters=MelBank(fs,512,40);

[FFT_power,Melbank_fft,MFCC,Feature_vector]=get_FFTP_Melbank_MFCC_Featurevector(frames,512,Mel_filters);


%plotting the results-----------------------------------------------
%-------------------------------------------------------------------
% Create a figure with all subplots
figure('Position', [100, 100, 1400, 900]);

% 1. FFT Power Spectrum (Linear)
subplot(3,2,1);
imagesc(FFT_power);
colorbar;
title('FFT Power Spectrum (Linear)');
xlabel('Frame Number');
ylabel('Frequency Bin');
colormap(gca, 'jet');

% 2. FFT Power Spectrum (Log/dB)
subplot(3,2,2);
imagesc(10*log10(FFT_power+eps));
colorbar;
title('FFT Power Spectrum (dB)');
xlabel('Frame Number');
ylabel('Frequency Bin');
colormap(gca, 'jet');

% 3. Melbank Filtered FFT (Linear)
subplot(3,2,3);
imagesc(Melbank_fft);
colorbar;
title('Mel-bank Filtered Spectrogram (Linear)');
xlabel('Frame Number');
ylabel('Mel Filter Number');
colormap(gca, 'jet');

% 4. Melbank Filtered FFT (Log/dB)
subplot(3,2,4);
imagesc(10*log10(Melbank_fft+eps));
colorbar;
title('Mel-bank Filtered Spectrogram (dB)');
xlabel('Frame Number');
ylabel('Mel Filter Number');
colormap(gca, 'jet');

% 5. MFCC Coefficients
subplot(3,2,5);
imagesc(MFCC);
colorbar;
title('MFCC Coefficients');
xlabel('Frame Number');
ylabel('MFCC Coefficient');
colormap(gca, 'jet');

% 6. Final Feature Vector
subplot(3,2,6);
imagesc(Feature_vector);
colorbar;
title('Final Feature Vector (MFCC + Δ + ΔΔ)');
xlabel('Frame Number');
ylabel('Feature Index');
colormap(gca, 'jet');

% Create a separate figure for Mel-bank filters
figure('Position', [100, 100, 1200, 600]);

% Plot Mel-bank filters as individual curves
subplot(1,2,1);
plot(Mel_filters');
title('Mel-bank Filter Responses');
xlabel('Frequency Bin');
ylabel('Filter Magnitude');
grid on;
legend('Location', 'best');

% Plot Mel-bank filters as image/heatmap
subplot(1,2,2);
imagesc(Mel_filters);
colorbar;
title('Mel-bank Filters (Heatmap)');
xlabel('Frequency Bin');
ylabel('Mel Filter Number');
colormap(gca, 'hot');