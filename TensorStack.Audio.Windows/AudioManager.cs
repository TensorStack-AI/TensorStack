using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using TensorStack.Common;
using TensorStack.Common.Common;
using TensorStack.Common.Tensor;

namespace TensorStack.Audio.Windows
{
    public static class AudioManager
    {
        private static string FFMpegPath = "ffmpeg.exe";
        private static string FFProbePath = "ffprobe.exe";
        private static string DirectoryTemp = "Temp";

        /// <summary>
        /// Configures the specified ffmpeg/ffprobe path.
        /// </summary>
        /// <param name="ffmpegPath">The ffmpeg path.</param>
        /// <param name="ffprobePath">The ffprobe path.</param>
        /// <param name="directoryTemp">The directory temporary.</param>
        public static void Initialize(string ffmpegPath, string ffprobePath, string directoryTemp)
        {
            FFMpegPath = ffmpegPath;
            FFProbePath = ffprobePath;
            DirectoryTemp = directoryTemp;
        }


        /// <summary>
        /// Loads the audio information.
        /// </summary>
        /// <param name="filename">The filename.</param>
        public static AudioInfo LoadInfo(string filename)
        {
            return ReadInfo(filename);
        }


        /// <summary>
        /// Loads the audio information asynchronously.
        /// </summary>
        /// <param name="filename">The filename.</param>
        public static async Task<AudioInfo> LoadInfoAsync(string filename)
        {
            return await ReadInfoAsync(filename);
        }


        /// <summary>
        /// Loads the tensor.
        /// </summary>
        /// <param name="filename">The filename.</param>
        /// <param name="audioCodec">The audio codec.</param>
        /// <param name="sampleRate">The sample rate.</param>
        /// <param name="channels">The channels.</param>
        public static AudioTensor LoadTensor(string filename, string audioCodec = "pcm_s16le", int sampleRate = 16000, int channels = 1)
        {
            return ReadAudio(filename, audioCodec, sampleRate, channels);
        }


        /// <summary>
        /// Loads the tensor asynchronously.
        /// </summary>
        /// <param name="filename">The filename.</param>
        /// <param name="audioCodec">The audio codec.</param>
        /// <param name="sampleRate">The sample rate.</param>
        /// <param name="channels">The channels.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        public static async Task<AudioTensor> LoadTensorAsync(string filename, string audioCodec = "pcm_s16le", int sampleRate = 16000, int channels = 1, CancellationToken cancellationToken = default)
        {
            return await ReadAudioAsync(filename, audioCodec, sampleRate, channels, cancellationToken);
        }


        /// <summary>
        /// Saves the audio to file.
        /// </summary>
        /// <param name="filename">The filename.</param>
        /// <param name="audioTensor">The audio tensor.</param>
        public static void SaveAudio(string filename, AudioTensor audioTensor)
        {
            WriteAudio(filename, audioTensor);
        }


        /// <summary>
        /// Saves the audio to file asynchronously.
        /// </summary>
        /// <param name="filename">The filename.</param>
        /// <param name="audioTensor">The audio tensor.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        public static async Task SaveAudioAync(string filename, AudioTensor audioTensor, CancellationToken cancellationToken = default)
        {
            await WriteAudioAsync(filename, audioTensor, cancellationToken);
        }


        /// <summary>
        /// Adds the audio from source video to target video.
        /// </summary>
        /// <param name="targetVideoFile">The target video file.</param>
        /// <param name="sourceVideoFile">The source video file.</param>
        public static void AddAudio(string targetVideoFile, string sourceVideoFile)
        {
            MuxAudio(targetVideoFile, sourceVideoFile);
        }


        /// <summary>
        /// Adds the audio from source video to target video asynchronously.
        /// </summary>
        /// <param name="targetVideoFile">The target video file.</param>
        /// <param name="sourceVideoFile">The source video file.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        public static async Task AddAudioAsync(string targetVideoFile, string sourceVideoFile, CancellationToken cancellationToken = default)
        {
            await MuxAudioAsync(targetVideoFile, sourceVideoFile, cancellationToken);
        }


        /// <summary>
        /// Reads the audio data as AudioTensor
        /// </summary>
        /// <param name="audioInputFile">The audio input file.</param>
        /// <param name="audioCodec">The audio codec.</param>
        /// <param name="sampleRate">The sample rate.</param>
        /// <param name="channels">The channels.</param>
        /// <returns>AudioTensor.</returns>
        private static AudioTensor ReadAudio(string audioInputFile, string audioCodec, int sampleRate, int channels)
        {
            using (var ffmpeg = CreateReader(audioInputFile, audioCodec, sampleRate, channels))
            using (var audioStream = new MemoryStream())
            {
                ffmpeg.Start();
                ffmpeg.StandardOutput.BaseStream.CopyTo(audioStream);
                ffmpeg.WaitForExit();

                var audioBytes = audioStream.ToArray();
                return CreateAudioTensor(audioBytes, channels, sampleRate);
            }
        }


        /// <summary>
        /// Reads the audio data as AudioTensor asynchronously.
        /// </summary>
        /// <param name="audioInputFile">The audio input file.</param>
        /// <param name="audioCodec">The audio codec.</param>
        /// <param name="sampleRate">The sample rate.</param>
        /// <param name="channels">The channels.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        private static async Task<AudioTensor> ReadAudioAsync(string audioInputFile, string audioCodec, int sampleRate, int channels, CancellationToken cancellationToken = default)
        {
            using (var ffmpeg = CreateReader(audioInputFile, audioCodec, sampleRate, channels))
            using (var audioStream = new MemoryStream())
            {
                ffmpeg.Start();
                await ffmpeg.StandardOutput.BaseStream.CopyToAsync(audioStream, cancellationToken);
                await ffmpeg.WaitForExitAsync(cancellationToken);

                var audioBytes = audioStream.ToArray();
                return CreateAudioTensor(audioBytes, channels, sampleRate);
            }
        }


        /// <summary>
        /// Writes the AudioTensor to file.
        /// </summary>
        /// <param name="audioOutputFile">The audio output file.</param>
        /// <param name="audioTensor">The audio tensor.</param>
        private static void WriteAudio(string audioOutputFile, AudioTensor audioTensor)
        {
            var samples = audioTensor.Samples;
            var channels = audioTensor.Channels;
            var sampleRate = audioTensor.SampleRate;
            using (var ffmpeg = CreateWriter(audioOutputFile, sampleRate, channels))
            {
                ffmpeg.Start();
                using (var audioStream = ffmpeg.StandardInput.BaseStream)
                {
                    var audioBuffer = CreateAudioBuffer(audioTensor, channels, samples);
                    audioStream.Write(audioBuffer);
                    audioStream.Flush();
                }
                ffmpeg.WaitForExit();
            }
        }


        /// <summary>
        /// Writes the AudioTensor to file asynchronously.
        /// </summary>
        /// <param name="audioOutputFile">The audio output file.</param>
        /// <param name="audioTensor">The audio tensor.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        private static async Task WriteAudioAsync(string audioOutputFile, AudioTensor audioTensor, CancellationToken cancellationToken = default)
        {
            var samples = audioTensor.Samples;
            var channels = audioTensor.Channels;
            var sampleRate = audioTensor.SampleRate;
            using (var ffmpeg = CreateWriter(audioOutputFile, sampleRate, channels))
            {
                ffmpeg.Start();
                using (var audioStream = ffmpeg.StandardInput.BaseStream)
                {
                    var audioBuffer = CreateAudioBuffer(audioTensor, channels, samples);
                    await audioStream.WriteAsync(audioBuffer, cancellationToken);
                    await audioStream.FlushAsync(cancellationToken);
                }
                await ffmpeg.WaitForExitAsync(cancellationToken);
            }
        }


        /// <summary>
        /// Muxes the audio.
        /// </summary>
        /// <param name="targetVideoFile">The target video file.</param>
        /// <param name="sourceVideoFile">The source video file.</param>
        internal static void MuxAudio(string targetVideoFile, string sourceVideoFile)
        {
            var tempFile = FileHelper.RandomFileName(DirectoryTemp, targetVideoFile);

            try
            {
                using (var ffmpeg = CreateMuxer(targetVideoFile, sourceVideoFile, tempFile))
                {
                    ffmpeg.Start();
                    ffmpeg.WaitForExit();
                }

                if (File.Exists(tempFile))
                    File.Move(tempFile, targetVideoFile, true);
            }
            finally
            {
                FileHelper.DeleteFile(tempFile);
            }
        }


        /// <summary>
        /// Muxes the audio asynchronously.
        /// </summary>
        /// <param name="targetVideoFile">The target video file.</param>
        /// <param name="sourceVideoFile">The source video file.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        internal static async Task MuxAudioAsync(string targetVideoFile, string sourceVideoFile, CancellationToken cancellationToken = default)
        {
            var tempFile = FileHelper.RandomFileName(DirectoryTemp, targetVideoFile);

            try
            {
                using (var ffmpeg = CreateMuxer(targetVideoFile, sourceVideoFile, tempFile))
                {
                    ffmpeg.Start();
                    await ffmpeg.WaitForExitAsync(cancellationToken);
                }

                if (File.Exists(tempFile))
                    File.Move(tempFile, targetVideoFile, true);
            }
            finally
            {
                FileHelper.DeleteFile(tempFile);
            }
        }


        /// <summary>
        /// Reads the information.
        /// </summary>
        /// <param name="filename">The filename.</param>
        /// <returns>AudioInfo.</returns>
        internal static AudioInfo ReadInfo(string filename)
        {
            using (var metadataReader = CreateMetadata(filename))
            {
                metadataReader.Start();
                var videoInfo = default(AudioInfo);
                using (var reader = metadataReader.StandardOutput)
                {
                    videoInfo = ParseInfo(filename, reader.ReadToEnd());
                }
                metadataReader.WaitForExit();
                return videoInfo;
            }
        }


        /// <summary>
        /// Reads the information asynchronously.
        /// </summary>
        /// <param name="filename">The filename.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        internal static async Task<AudioInfo> ReadInfoAsync(string filename, CancellationToken cancellationToken = default)
        {
            using (var metadataReader = CreateMetadata(filename))
            {
                metadataReader.Start();
                var videoInfo = default(AudioInfo);
                using (var reader = metadataReader.StandardOutput)
                {
                    videoInfo = ParseInfo(filename, await reader.ReadToEndAsync(cancellationToken));
                }
                await metadataReader.WaitForExitAsync(cancellationToken);
                return videoInfo;
            }
        }


        /// <summary>
        /// Parses the information.
        /// </summary>
        /// <param name="filename">The filename.</param>
        /// <param name="jsonString">The json string.</param>
        /// <returns>AudioInfo.</returns>
        /// <exception cref="System.Exception">Failed to parse audio stream metadata</exception>
        private static AudioInfo ParseInfo(string filename, string jsonString)
        {
            var metadata = JsonSerializer.Deserialize<AudioMetadata>(jsonString);
            var stream = metadata.Streams.FirstOrDefault(x => x.Type == "audio");
            if (stream is null)
                throw new Exception("Failed to parse audio stream metadata");

            return new AudioInfo
            {
                FileName = filename,
                AudioCodec = stream.CodecName,
                Channels = stream.Channels,
                SampleRate = stream.SampleRate,
                Samples = stream.SampleCount,
                Duration = stream.Duration,
            };
        }


        /// <summary>
        /// Creates the audio buffer.
        /// </summary>
        /// <param name="audioTensor">The audio tensor.</param>
        /// <param name="channels">The channels.</param>
        /// <param name="samples">The samples.</param>
        /// <returns>System.Byte[].</returns>
        private static byte[] CreateAudioBuffer(AudioTensor audioTensor, int channels, int samples)
        {
            int offset = 0;
            byte[] buffer = new byte[samples * channels * 4]; // float32 = 4 bytes
            for (int i = 0; i < samples; i++)
            {
                for (int c = 0; c < channels; c++)
                {
                    float sample = Math.Clamp(audioTensor[c, i], -1f, 1f);
                    byte[] bytes = BitConverter.GetBytes(sample);
                    Buffer.BlockCopy(bytes, 0, buffer, offset, 4);
                    offset += 4;
                }
            }
            return buffer;
        }


        /// <summary>
        /// Creates the audio tensor.
        /// </summary>
        /// <param name="audioBytes">The audio bytes.</param>
        /// <param name="channels">The channels.</param>
        /// <param name="sampleRate">The sample rate.</param>
        /// <returns>AudioTensor.</returns>
        private static AudioTensor CreateAudioTensor(byte[] audioBytes, int channels, int sampleRate)
        {
            // Convert PCM16 -> float32 [-1, 1]
            var sampleCount = audioBytes.Length / 2 / channels;
            var result = new Tensor<float>([channels, sampleCount]);
            for (int i = 0, s = 0; i < audioBytes.Length; i += 2, s++)
            {
                short sample = BitConverter.ToInt16(audioBytes, i);
                float normalized = sample / 32768f;

                int channel = s % channels;
                int frame = s / channels;
                result[channel, frame] = normalized;
            }

            return result.AsAudioTensor(sampleRate);
        }


        #region FFMPEG / FFProbe

        private static Process CreateProcess(string executable, string arguments)
        {
            var ffmpegProcess = new Process();
            ffmpegProcess.StartInfo.FileName = executable;
            ffmpegProcess.StartInfo.Arguments = arguments;
            ffmpegProcess.StartInfo.UseShellExecute = false;
            ffmpegProcess.StartInfo.CreateNoWindow = true;
            return ffmpegProcess;
        }


        private static Process CreateReader(string inputFile, string audioCodec, int sampleRate, int channels)
        {
            var process = CreateProcess(FFMpegPath, $"-hide_banner -i \"{inputFile}\" -f s16le -acodec {audioCodec} -ac {channels} -ar {sampleRate} pipe:1");
            process.StartInfo.RedirectStandardOutput = true;
            return process;
        }


        private static Process CreateWriter(string audioOutputFile, int sampleRate, int channels)
        {
            var process = CreateProcess(FFMpegPath, $"-hide_banner -y -f f32le -ac {channels} -ar {sampleRate} -i pipe:0 \"{audioOutputFile}\"");
            process.StartInfo.RedirectStandardInput = true;
            return process;
        }


        private static Process CreateMuxer(string targetVideo, string sourceVideo, string tempFile)
        {
            var process = CreateProcess(FFMpegPath, $"-hide_banner -i \"{targetVideo}\" -i \"{sourceVideo}\" -c:v copy -c:a copy -map 0:v:0 -map 1:a:0 -y \"{tempFile}\"");
            process.StartInfo.RedirectStandardInput = true;
            return process;
        }


        private static Process CreateMetadata(string inputFile)
        {
            var process = CreateProcess(FFProbePath, $"-v quiet -print_format json -show_format -show_streams {inputFile}");
            process.StartInfo.RedirectStandardInput = true;
            process.StartInfo.RedirectStandardOutput = true;
            return process;
        }


        private record AudioMetadata
        {
            [JsonPropertyName("format")]
            public AudioFormat Format { get; set; }

            [JsonPropertyName("streams")]
            public List<AudioStream> Streams { get; set; }
        }

        [JsonNumberHandling(JsonNumberHandling.AllowReadingFromString)]
        private record AudioFormat
        {
            [JsonPropertyName("filename")]
            public string FileName { get; set; }

            [JsonPropertyName("nb_streams")]
            public int StreamCount { get; set; }

            [JsonPropertyName("format_name")]
            public string FormatName { get; set; }

            [JsonPropertyName("format_long_name")]
            public string FormatLongName { get; set; }

            [JsonPropertyName("size")]
            public long Size { get; set; }

            [JsonPropertyName("bit_rate")]
            public long BitRate { get; set; }
        }

        [JsonNumberHandling(JsonNumberHandling.AllowReadingFromString)]
        private record AudioStream
        {
            [JsonPropertyName("codec_type")]
            public string Type { get; set; }

            [JsonPropertyName("codec_name")]
            public string CodecName { get; set; }

            [JsonPropertyName("codec_long_name")]
            public string CodecLongName { get; set; }

            [JsonPropertyName("sample_fmt")]
            public string SampleFormat { get; set; }

            [JsonPropertyName("sample_rate")]
            public int SampleRate { get; set; }

            [JsonPropertyName("duration_ts")]
            public long SampleCount { get; set; }

            [JsonPropertyName("channels")]
            public int Channels { get; set; }

            [JsonPropertyName("bits_per_sample")]
            public int BitsPerSample { get; set; }

            [JsonPropertyName("bit_rate")]
            public int BitRate { get; set; }

            [JsonPropertyName("duration")]
            public float DurationSeconds { get; set; }

            public TimeSpan Duration => TimeSpan.FromSeconds(DurationSeconds);
        }

        #endregion
    }
}
