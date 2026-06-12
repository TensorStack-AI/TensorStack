using Amuse.App.Common;
using Amuse.Common;
using System;
using System.Buffers;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Windows;
using System.Windows.Media;
using TensorStack.Common;

namespace Amuse.App
{
    public static class Extensions
    {
        private static readonly SearchValues<char> InvalidPathChars = SearchValues.Create(Path.GetInvalidPathChars());

        /// <summary>
        /// Gets the file extesnion.
        /// </summary>
        /// <param name="mediaType">Type of the media.</param>
        public static string GetExtension(this MediaType mediaType)
        {
            return mediaType switch
            {
                MediaType.Text => "txt",
                MediaType.Audio => "wav",
                MediaType.Video => "mp4",
                MediaType.Image => "png",
                _ => throw new NotSupportedException()
            };
        }


        public static int GetIndex(this MemoryProfile profile, int deviceMemory)
        {
            int bestIndex = -1;
            int bestValue = int.MinValue;

            for (int i = 0; i < profile.MemoryModes.Length; i++)
            {
                int value = profile.MemoryModes[i];
                if (value <= deviceMemory && value > bestValue)
                {
                    bestValue = value;
                    bestIndex = i;
                }
            }

            if (bestIndex < 0)
                bestIndex = 0;

            return bestIndex;
        }


        public static bool HasChanged(this IReadOnlyList<LoraAdapterModel> existingAdapters, IReadOnlyList<LoraAdapterModel> newAdapters)
        {
            if (ReferenceEquals(existingAdapters, newAdapters))
                return false;

            if (existingAdapters == null || newAdapters == null)
                return true;

            if (existingAdapters.Count != newAdapters.Count)
                return true;

            for (int i = 0; i < existingAdapters.Count; i++)
            {
                if (!string.Equals(existingAdapters[i]?.Key, newAdapters[i]?.Key, StringComparison.Ordinal))
                    return true;
            }
            return false;
        }


        public static CheckpointConfig ToConfig(this CheckpointModel checkpoint, Settings settings)
        {
            var modelDirectory = settings.DirectoryDiffusion;
            var checkpointConfig = new CheckpointConfig
            {
                Compute = checkpoint.Compute?.Resolve(settings, modelDirectory),
                TextEncoder = checkpoint.TextEncoder?.Resolve(settings, modelDirectory),
                TextEncoder2 = checkpoint.TextEncoder2?.Resolve(settings, modelDirectory),
                TextEncoder3 = checkpoint.TextEncoder3?.Resolve(settings, modelDirectory),
                Unet = checkpoint.Unet?.Resolve(settings, modelDirectory),
                Transformer = checkpoint.Transformer?.Resolve(settings, modelDirectory),
                Transformer2 = checkpoint.Transformer2?.Resolve(settings, modelDirectory),
                Vae = checkpoint.Vae?.Resolve(settings, modelDirectory),
                AudioVae = checkpoint.AudioVae?.Resolve(settings, modelDirectory),
                Vocoder = checkpoint.Vocoder?.Resolve(settings, modelDirectory),
                Connectors = checkpoint.Connectors?.Resolve(settings, modelDirectory),
                LatentUpsampler = checkpoint.LatentUpsampler?.Resolve(settings, modelDirectory),
                LatentUpsamplerTemporal = checkpoint.LatentUpsamplerTemporal?.Resolve(settings, modelDirectory),
                ConditionEncoder = checkpoint.ConditionEncoder?.Resolve(settings, modelDirectory),
                AudioTokenizer = checkpoint.AudioTokenizer?.Resolve(settings, modelDirectory),
                AudioDetokenizer = checkpoint.AudioDetokenizer?.Resolve(settings, modelDirectory),
            };
            return checkpointConfig;
        }


        public static SchedulerInputOptions[] Copy(this SchedulerInputOptions[] collection)
        {
            if (collection.IsNullOrEmpty())
                return null;

            return collection.Select(x => x with
            {
                ScaleFactors = x.ScaleFactors?.ToList(),
                StageRange = x.StageRange?.ToList(),
                DisableCorrector = x.DisableCorrector?.ToList(),
            }).ToArray();
        }


        public static MemoryProfile[] Copy(this MemoryProfile[] collection)
        {
            if (collection.IsNullOrEmpty())
                return [];

            return collection.Select(x => new MemoryProfile
            {
                QualityMode = x.QualityMode,
                MemoryModes = x.MemoryModes.ToArray(),
            }).ToArray();
        }


        public static SizeOption[] Copy(this SizeOption[] collection)
        {
            if (collection.IsNullOrEmpty())
                return [];

            return collection.Select(x => new SizeOption
            {
                Height = x.Height,
                Width = x.Width,
                IsDefault = x.IsDefault
            }).ToArray();
        }


        public static List<LoraConfig> GetLoraAdapters(this LoraAdapterModel[] loraAdapterModel, Settings settings)
        {
            if (loraAdapterModel.IsNullOrEmpty())
                return default;

            var loraConfigs = new List<LoraConfig>();
            var modelDirectory = settings.DirectoryLoraAdapter;
            foreach (var loraAdapter in loraAdapterModel)
            {
                var resolvedCheckpoint = loraAdapter.Checkpoint?.Resolve(settings, modelDirectory);
                var loraPath = Path.GetDirectoryName(resolvedCheckpoint);
                var loraWeights = Path.GetFileName(resolvedCheckpoint);
                loraConfigs.Add(new LoraConfig
                {
                    Path = loraPath,
                    Weights = loraWeights,
                    Name = loraAdapter.Key
                });
            }

            return loraConfigs;
        }


        public static List<LoraOptions> GetLoraOptions(this DiffusionInputOptions options)
        {
            if (options.LoraOptions.IsNullOrEmpty())
                return default;

            return [.. options.LoraOptions.Select(x => new LoraOptions
            {
                Name = x.Key,
                Strength = x.Strength
            })];
        }


        public static ControlNetConfig GetControlNet(this ControlNetModel model, Settings settings)
        {
            if (model is null)
                return null;

            var resolvedCheckpoint = model.Checkpoint.Resolve(settings, settings.DirectoryControlNet);
            return new ControlNetConfig
            {
                Name = model.Name,
                Path = resolvedCheckpoint,
                Invert = model.Invert,
                LayerCount = model.LayerCount,
                DisableProjections = model.DisableProjections
            };
        }


        public static PipelineLoadOptions ToClientOptions(this PipelineModel pipelineConfig, Settings settings)
        {
            var device = pipelineConfig.Device;
            var model = pipelineConfig.DiffusionModel;
            var controlNet = pipelineConfig.ControlNetModel;
            return new PipelineLoadOptions
            {
                Variant = model.Variant,
                ModelPath = Path.GetFullPath(settings.DirectoryDiffusion),
                Template = model.Template,
                Pipeline = model.Pipeline.ToString(),
                ModelType = model.ModelType,
                ProcessType = pipelineConfig.ProcessType,
                Device = device.DeviceCode,
                DeviceId = device.DeviceId,
                DeviceBusId = device.PCIBusId,
                DataType = model.BaseType,
                IsOptimizeDeviceEnabled = settings.IsOptimizeDeviceEnabled,
                IsOptimizeChannelsEnabled = settings.IsOptimizeChannelsEnabled,
                IsDeviceQuantizationEnabled = settings.IsDeviceQuantizationEnabled,
                MemoryMode = pipelineConfig.GetMemoryMode(),
                QuantType = pipelineConfig.GetQuantizationType(),

                ControlNet = controlNet.GetControlNet(settings),
                LoraAdapters = pipelineConfig.LoraAdapterModel.GetLoraAdapters(settings),
                CheckpointConfig = model.Checkpoint.ToConfig(settings)
            };
        }


        private static MemoryModeType GetMemoryMode(this PipelineModel pipeline)
        {
            var memoryMode = pipeline.MemoryMode;
            if (memoryMode == MemoryMode.Auto)
            {
                var memoryProfile = pipeline.DiffusionModel.MemoryProfile.FirstOrDefault(x => x.QualityMode == pipeline.QualityMode);
                if (memoryProfile != null)
                {
                    var deviceMemory = pipeline.Device.MemoryGB;
                    var modeIndex = memoryProfile.GetIndex(deviceMemory);
                    memoryMode = Enum.GetValues<MemoryMode>()[modeIndex + 2];
                }
            }

            return memoryMode switch
            {
                MemoryMode.Balanced => MemoryModeType.Balanced,
                MemoryMode.Low => MemoryModeType.OffloadCPU,
                MemoryMode.Medium => MemoryModeType.OffloadModel,
                MemoryMode.High => MemoryModeType.Device,
                _ => MemoryModeType.OffloadCPU,
            };
        }


        private static QuantizationType GetQuantizationType(this PipelineModel pipeline)
        {
            return pipeline.QualityMode switch
            {
                QualityMode.Draft => QuantizationType.Q4Bit,
                QualityMode.Standard => QuantizationType.Q8Bit,
                QualityMode.Production => QuantizationType.Q16Bit,
                _ => QuantizationType.Q8Bit,
            };
        }


        public static PipelineCreateOptions ToClientOptions(this EnvironmentModel environment, Settings settings, EnvironmentMode environmentMode)
        {
            var environmentConfig = new PipelineCreateOptions
            {
                IsDebug = settings.IsServerDebugEnabled,
                Directory = App.DirectoryPython,
                Environment = environment.Environment,
                PythonVersion = environment.PythonVersion,
                Requirements = environment.Requirements.ToArray(),
                Variables = environment.Variables?.ToDictionary() ?? new Dictionary<string, string>(),
                Mode = environmentMode
            };

            environmentConfig.Variables.Add("HF_HUB_OFFLINE", "1");
            environmentConfig.Variables.Add("HF_HUB_CACHE", settings.DirectoryDiffusion);
            return environmentConfig;
        }


        public static PipelineRunOptions ToClientOptions(this DiffusionInputOptions options, DiffusionDefaultOptions defaultOptions, string tempFileName)
        {
            return new PipelineRunOptions
            {
                Seed = options.Seed,
                Steps = options.Steps,
                Steps2 = options.Steps2,
                GuidanceScale = options.GuidanceScale,
                GuidanceScale2 = options.GuidanceScale2,
                Prompt = options.Prompt,
                Prompt2 = options.Prompt2,
                NegativePrompt = options.NegativePrompt,
                Strength = options.Strength,
                Duration = options.Duration,
                Bpm = options.Bpm,
                Instruction = options.Instruction,
                Keyscale = options.Keyscale,
                MaxLength = defaultOptions.MaxLength,
                MaxLength2 = defaultOptions.MaxLength2,
                Task = options.Task,
                TimeSignature = options.TimeSignature,
                TrackName = options.TrackName,
                TempFileName = tempFileName,
                EnableVaeSlicing = options.IsVaeSlicingEnabled,
                EnableVaeTiling = options.IsVaeTilingEnabled,
                SchedulerOptions = options.SchedulerOptions?.ToClientOptions(),
                LoraOptions = options.GetLoraOptions(),

                Beams = options.Beams,
                TopK = options.TopK,
                TopP = options.TopP,
                Temperature = options.Temperature,
                MinLength = options.MinLength,
                NoRepeatNgramSize = options.NoRepeatNgramSize,
                LengthPenalty = options.LengthPenalty,
                DiversityLength = options.DiversityLength,
                EarlyStopping = options.EarlyStopping.ToString(),
                Language = options.Language,
                ChunkSize = options.ChunkSize,

                ControlNetScale = options.ControlNetStrength,
                FrameChunk = options.FrameChunk,
                FrameChunkOverlap = options.FrameChunkOverlap,
                FrameRate = options.FrameRate,
                Frames = options.Frames,
                Height = options.Height,
                NoiseCondition = options.NoiseCondition,
                SilenceDuration = options.SilenceDuration,
                Speed = options.Speed,
                Width = options.Width,

                SampleRate = defaultOptions.SampleRate,

                InputImages = options.InputImages,
                InputControlImages = options.InputControlImages
            };
        }




        public static bool IsValidPath(string path)
        {
            if (string.IsNullOrWhiteSpace(path))
                return false;

            return path.AsSpan().IndexOfAny(InvalidPathChars) == -1;
        }


        public static bool AddIfNotNull<TSource>(this IList<TSource> source, TSource item)
        {
            if (item is null)
                return false;

            source.Add(item);
            return true;
        }


        public static int RemoveAll<T>(this IList<T> collection, Predicate<T> condition)
        {
            var removed = 0;
            for (int i = collection.Count - 1; i >= 0; i--)
            {
                if (condition(collection[i]))
                {
                    collection.RemoveAt(i);
                    removed++;
                }
            }
            return removed;
        }
    }

    public static partial class Utils
    {
        public const int FixedIdRange = 1000;
    }



    public static class FontOptions
    {
        public static FontWeight[] FontWeightList { get; } = new[]
           {
            FontWeights.Thin,
            FontWeights.ExtraLight,
            FontWeights.Light,
            FontWeights.Normal,
            FontWeights.Medium,
            FontWeights.SemiBold,
            FontWeights.Bold,
            FontWeights.ExtraBold,
            FontWeights.Black
        };


        public static FontStyle[] FontStyleList { get; } = new[]
        {
            FontStyles.Normal,
            FontStyles.Italic,
            FontStyles.Oblique
        };


        public static ICollection<FontFamily> FontFamilies { get; } = System.Windows.Media.Fonts.SystemFontFamilies;
    }


    public static class BrushOptions
    {
        public static IEnumerable<Brush> AllBrushes { get; } =
            typeof(Brushes).GetProperties()
                .Where(p => p.PropertyType == typeof(Brush))
                .Select(p => (Brush)p.GetValue(null));
    }
}
