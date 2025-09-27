using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using TensorStack.Common.Tensor;

namespace TensorStack.TextGeneration.Pipelines.Whisper
{
    public class WhisperPreprocessor
    {
        private readonly float[] _melFilterBank;
        private readonly int _sampleRate  = 16000;
        private readonly int _frameSize = 400;    // 25ms @ 16kHz
        private readonly int _hopLength  = 160;   // 10ms @ 16kHz
        private readonly int _numMelBins  = 80;
        private readonly int _nfft  = 512;

        public WhisperPreprocessor()
        {
            _melFilterBank = CreateMelFilterBank();
        }


        public Tensor<float> Process(string wavPath)
        {
            var samples = LoadWavPcm16Mono(wavPath);

            // Pre-emphasis (optional, Whisper doesn’t strictly need it)
            // for (int i = samples.Length - 1; i > 0; i--)
            //     samples[i] -= 0.97f * samples[i - 1];

            var stft = STFT(samples);
            var melSpec = ApplyMel(stft);

            // log10(mel + epsilon)
            var result = new Tensor<float>([1, melSpec.GetLength(0), melSpec.GetLength(1)]);
            for (int i = 0; i < melSpec.GetLength(0); i++)
                for (int j = 0; j < melSpec.GetLength(1); j++)
                    result[0, i, j] = (float)Math.Log10(Math.Max(1e-10f, melSpec[i, j]));

            return result; 
        }


        private float[] LoadWavPcm16Mono(string path)
        {
            using var br = new BinaryReader(File.OpenRead(path));
            br.ReadBytes(44); // skip WAV header
            var data = new List<float>();
            while (br.BaseStream.Position < br.BaseStream.Length)
                data.Add(br.ReadInt16() / 32768f);
            return data.ToArray();
        }


        private Complex[][] STFT(float[] samples)
        {
            int numFrames = 1 + (samples.Length - _frameSize) / _hopLength;
            var frames = new Complex[numFrames][];

            // Hann window
            float[] window = Enumerable.Range(0, _frameSize)
                .Select(n => 0.5f - 0.5f * (float)Math.Cos(2 * Math.PI * n / _frameSize))
                .ToArray();

            for (int i = 0; i < numFrames; i++)
            {
                var frame = new Complex[_nfft];
                int start = i * _hopLength;
                for (int j = 0; j < _frameSize; j++)
                    frame[j] = samples[start + j] * window[j];
                for (int j = _frameSize; j < _nfft; j++)
                    frame[j] = Complex.Zero;

                FFT(frame); // in-place
                frames[i] = frame;
            }

            return frames;
        }


        private float[,] ApplyMel(Complex[][] stft)
        {
            int numFrames = stft.Length;
            float[,] melSpec = new float[_numMelBins, numFrames];

            for (int t = 0; t < numFrames; t++)
            {
                float[] power = new float[_nfft / 2 + 1];
                for (int f = 0; f < power.Length; f++)
                    power[f] = (float)(stft[t][f].Magnitude * stft[t][f].Magnitude);

                for (int m = 0; m < _numMelBins; m++)
                {
                    float sum = 0;
                    for (int f = 0; f < power.Length; f++)
                        sum += power[f] * _melFilterBank[m * power.Length + f];
                    melSpec[m, t] = sum;
                }
            }

            return melSpec;
        }


        private float[] CreateMelFilterBank()
        {
            int numFreqs = _nfft / 2 + 1;
            float[] filterBank = new float[_numMelBins * numFreqs];

            double fMin = 0;
            double fMax = _sampleRate / 2;
            double melMin = HzToMel(fMin);
            double melMax = HzToMel(fMax);

            double[] melPoints = Enumerable.Range(0, _numMelBins + 2)
                .Select(i => melMin + (melMax - melMin) * i / (_numMelBins + 1))
                .ToArray();

            double[] hzPoints = melPoints.Select(MelToHz).ToArray();
            int[] bins = hzPoints.Select(hz => (int)Math.Floor((_nfft + 1) * hz / _sampleRate)).ToArray();

            for (int m = 1; m <= _numMelBins; m++)
            {
                int f0 = bins[m - 1], f1 = bins[m], f2 = bins[m + 1];
                for (int f = f0; f < f1; f++)
                    filterBank[(m - 1) * numFreqs + f] = (float)(f - f0) / (f1 - f0);
                for (int f = f1; f < f2; f++)
                    filterBank[(m - 1) * numFreqs + f] = (float)(f2 - f) / (f2 - f1);
            }

            return filterBank;
        }

        private static double HzToMel(double hz) => 2595 * Math.Log10(1 + hz / 700);
        private static double MelToHz(double mel) => 700 * (Math.Pow(10, mel / 2595) - 1);


        private void FFT(Complex[] buffer)
        {
            int n = buffer.Length;
            int bits = (int)Math.Log2(n);

            // bit-reversal
            for (int i = 1, j = 0; i < n; i++)
            {
                int bit = n >> 1;
                for (; (j & bit) != 0; bit >>= 1) j ^= bit;
                j ^= bit;
                if (i < j)
                {
                    var temp = buffer[i];
                    buffer[i] = buffer[j];
                    buffer[j] = temp;
                }
            }

            // FFT
            for (int len = 2; len <= n; len <<= 1)
            {
                double ang = -2 * Math.PI / len;
                Complex wlen = new Complex(Math.Cos(ang), Math.Sin(ang));
                for (int i = 0; i < n; i += len)
                {
                    Complex w = Complex.One;
                    for (int j = 0; j < len / 2; j++)
                    {
                        Complex u = buffer[i + j];
                        Complex v = buffer[i + j + len / 2] * w;
                        buffer[i + j] = u + v;
                        buffer[i + j + len / 2] = u - v;
                        w *= wlen;
                    }
                }
            }
        }
    }
}
