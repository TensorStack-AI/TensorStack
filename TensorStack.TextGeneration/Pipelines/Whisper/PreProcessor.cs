using MathNet.Numerics.IntegralTransforms;
using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using TensorStack.Common.Tensor;

namespace TensorStack.TextGeneration.Pipelines.Whisper
{
    public class PreProcessor
    {
        private readonly int _nfft = 400;
        private readonly int _nFreqs = 201;
        private readonly int _numMels = 80;
        private readonly int _hopLength = 160;
        private readonly int _windowLength = 400;
        private readonly int _sampleRate = 16000;
        private readonly Matrix<float> _melFilters;
        private readonly float[] _window;

        /// <summary>
        /// Initializes a new instance of the <see cref="PreProcessor"/> class.
        /// </summary>
        /// <param name="melFilterPath">The MelFilters path</param>
        public PreProcessor(string melFilterPath)
        {
            _window = HannWindow(_windowLength);
            _melFilters = LoadMelFilters(melFilterPath);
        }


        /// <summary>
        /// Processes the specified input audio.
        /// </summary>
        /// <param name="inputAudio">The input audio.</param>
        /// <param name="nFrames">The n frames.</param>
        /// <returns>Tensor&lt;System.Single&gt;[].</returns>
        public Tensor<float>[] ProcessInput(Tensor<float> inputAudio, int nFrames = 3000)
        {
            var audioData = inputAudio.Span;
            int totalFrames = 1 + (audioData.Length + _nfft - 1) / _hopLength;

            var stft = STFT(audioData, totalFrames);
            var melSpec = MelSpectrogram(stft);

            // Convert to batches of [1, numMels, nFrames]
            int numBatches = (int)Math.Ceiling((double)melSpec.RowCount / nFrames);
            var batches = new Tensor<float>[numBatches];
            for (int b = 0; b < numBatches; b++)
            {
                var batch = new Tensor<float>([1, _numMels, nFrames]);
                for (int t = 0; t < nFrames; t++)
                {
                    int srcRow = b * nFrames + t;
                    if (srcRow >= melSpec.RowCount)
                        break;

                    for (int m = 0; m < _numMels; m++)
                        batch[0, m, t] = melSpec[srcRow, m];
                }
                batches[b] = batch;
            }
            return batches;
        }


        /// <summary>
        /// Runs the Mel spectrogram.
        /// </summary>
        /// <param name="stft">The STFT.</param>
        /// <returns>Matrix&lt;System.Single&gt;.</returns>
        private Matrix<float> MelSpectrogram(float[,] stft)
        {
            var magMatrix = Matrix<float>.Build.DenseOfArray(stft);  // [frames, nFreqs]
            var melSpec = magMatrix * _melFilters;                         // [frames, nMels]

            // Log10 + clip + normalize
            melSpec.MapInplace(x => MathF.Log10(MathF.Max(x, 1e-10f)));
            float maxVal = melSpec.Enumerate().Max();
            melSpec.MapInplace(x => MathF.Max(x, maxVal - 8.0f));
            melSpec.MapInplace(x => (x + 4.0f) / 4.0f);
            return melSpec;
        }


        /// <summary>
        /// Runs STFT
        /// </summary>
        /// <param name="audioData">The audio data.</param>
        /// <param name="totalFrames">The total frames.</param>
        /// <returns>System.Single[,].</returns>
        private float[,] STFT(ReadOnlySpan<float> audioData, int totalFrames)
        {
            var magnitudes = new float[totalFrames, _nfft / 2 + 1];
            for (int n = 0, i = 0; i < audioData.Length; i += _hopLength, n++)
            {
                var frame = new Complex[_nfft];
                for (int j = 0; j < _nfft; j++)
                {
                    int idx = i + j - _nfft / 2;
                    if (idx < 0) idx = -idx;
                    else if (idx >= audioData.Length) idx = audioData.Length - (idx - audioData.Length) - 1;

                    frame[j] = new Complex(audioData[idx] * _window[j], 0);
                }

                Fourier.Forward(frame, FourierOptions.Matlab);
                for (int f = 0; f < _nfft / 2 + 1; f++)
                    magnitudes[n, f] = (float)(frame[f].Magnitude * frame[f].Magnitude);
            }
            return magnitudes;
        }


        /// <summary>
        /// Loads the mel filters.
        /// </summary>
        /// <param name="path">The path.</param>
        /// <returns>Matrix&lt;System.Single&gt;.</returns>
        /// <exception cref="System.IO.InvalidDataException">Not a valid npy file</exception>
        private Matrix<float> LoadMelFilters(string melFilterPath)
        {
            if (!File.Exists(melFilterPath))
                throw new ArgumentException($"Whisper MelFilters required, (see: https://github.com/openai/whisper/tree/main/whisper/assets/Mel_filters.npz)");

            using (var zip = System.IO.Compression.ZipFile.OpenRead(melFilterPath))
            {
                var entry = zip.GetEntry($"mel_{_numMels}.npy"); // 80, 128
                using (var stream = entry.Open())
                using (var reader = new BinaryReader(stream))
                {
                    // Skip NPY header
                    var magic = reader.ReadBytes(6);
                    if (magic[0] != 0x93)
                        throw new InvalidDataException("Not a valid npy file");

                    reader.ReadBytes(2); // version
                    var headerLen = reader.ReadUInt16();
                    reader.ReadBytes(headerLen); // skip header

                    // Read floats
                    var buffer = new byte[4];
                    var floats = new List<float>();
                    while (reader.BaseStream.Read(buffer, 0, 4) == 4)
                        floats.Add(BitConverter.ToSingle(buffer, 0));

                    var matrix = Matrix<float>.Build.Dense(_nFreqs, _numMels);
                    for (int m = 0; m < _numMels; m++)
                        for (int f = 0; f < _nFreqs; f++)
                            matrix[f, m] = floats[m * _nFreqs + f];

                    return matrix;
                }
            }
        }


        /// <summary>
        /// Create Hann window.
        /// </summary>
        /// <param name="size">The size.</param>
        /// <returns>System.Single[].</returns>
        private static float[] HannWindow(int size)
        {
            var w = new float[size];
            for (int i = 0; i < size; i++)
                w[i] = 0.5f * (1 - MathF.Cos(2 * MathF.PI * i / (size - 1)));
            return w;
        }

    }
}
